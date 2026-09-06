"""
Headless Blender Python Script for Deterministic glTF 2.0 Export.
Usage:
    blender --background --python blender_export_headless.py -- input_scene.blend output_asset.gltf
"""

import sys
import os
import bpy

def export_gltf_deterministic(input_blend: str, output_gltf: str):
    # Clear default scene if creating fresh or load blend file
    if input_blend and os.path.exists(input_blend):
        bpy.ops.wm.open_mainfile(filepath=input_blend)
    
    # Sort objects by name for deterministic export ordering
    for obj in sorted(bpy.data.objects, key=lambda x: x.name):
        if obj.type == 'MESH':
            # Ensure modifiers are evaluated consistently
            obj.select_set(True)

    # Export glTF 2.0 with explicit, deterministic parameters
    bpy.ops.export_scene.gltf(
        filepath=output_gltf,
        export_format='GLTF_EMBEDDED', # or GLTF_SEPARATE / GLB
        export_copyright="Type0 Hermetic Engine",
        export_image_format='AUTO',
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials='EXPORT',
        export_colors=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True, # Vulkan / standard Y-up conversion
        export_apply=True, # Apply modifiers deterministically
        export_attributes=True
    )
    print(f"[Blender Headless Export] Successfully exported deterministic glTF: {output_gltf}")

if __name__ == "__main__":
    # Parsing custom args passed after '--'
    args = sys.argv
    if "--" in args:
        custom_args = args[args.index("--") + 1:]
        if len(custom_args) >= 2:
            input_file = custom_args[0]
            output_file = custom_args[1]
            export_gltf_deterministic(input_file, output_file)
        else:
            print("Usage: blender --background --python blender_export_headless.py -- <input.blend> <output.gltf>")
    else:
        print("No input arguments supplied to script. Running standalone check.")
