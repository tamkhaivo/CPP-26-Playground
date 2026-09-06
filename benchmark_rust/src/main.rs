use rayon::prelude::*;
use std::time::Instant;
use wide::f32x8;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct MeshInstanceData {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

fn parallel_transform_pipeline(data_buffer: &mut [MeshInstanceData], delta_time: f32) {
    let chunk_size = 1024;
    
    data_buffer.par_chunks_mut(chunk_size).for_each(|chunk| {
        let simd_delta = f32x8::splat(delta_time);
        let velocity_mod = f32x8::splat(9.81);
        
        let mut i = 0;
        let lanes = 8;
        
        while i + lanes <= chunk.len() {
            // Load y values
            let y_values = f32x8::new([
                chunk[i].y, chunk[i+1].y, chunk[i+2].y, chunk[i+3].y,
                chunk[i+4].y, chunk[i+5].y, chunk[i+6].y, chunk[i+7].y
            ]);
            
            // Multiply and add
            let updated_y = (simd_delta * velocity_mod) + y_values;
            
            let arr = updated_y.to_array();
            for l in 0..lanes {
                chunk[i + l].y = arr[l];
            }
            
            i += lanes;
        }
        
        // Scalar fallback for tail
        while i < chunk.len() {
            chunk[i].y += delta_time * 9.81;
            i += 1;
        }
    });
}

fn main() {
    let num_objects = 10_000_000;
    let mut buffer = vec![MeshInstanceData { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }; num_objects];
    
    for (i, item) in buffer.iter_mut().enumerate() {
        item.y = (i % 100) as f32;
    }
    
    println!("Starting Rust Benchmark (Rayon + wide) with {} objects.", num_objects);
    
    // Warmup
    parallel_transform_pipeline(&mut buffer, 0.016);
    
    let start = Instant::now();
    let iterations = 100;
    
    for _ in 0..iterations {
        parallel_transform_pipeline(&mut buffer, 0.016);
    }
    
    let elapsed = start.elapsed();
    
    println!("Average time per frame: {} ms", elapsed.as_secs_f64() * 1000.0 / (iterations as f64));
    println!("Sample value: {}", buffer[500].y);
}
