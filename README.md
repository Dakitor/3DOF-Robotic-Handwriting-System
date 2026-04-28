# 3-DOF Robotic Handwriting System

This repository contains the software implementation and supplementary output files for the final year project:
Design and Implementation of a 3-DOF Robotic System for Human-like Handwriting.

## Main Files
- robust_centerline_gcode.py: PNG-to-G-code conversion program
- UI.py: PyQt user interface for text layout and combined G-code export
- gcode_library/: generated single-character G-code files
- sdt_generated_png_library/: SDT-generated PNG character images
- combined_gcode/: exported combined G-code files for demonstration

## Workflow
1. Generate personalised Chinese character PNG images using SDT.
2. Convert PNG images into single-character G-code files.
3. Load the G-code library in the PyQt UI.
4. Input text and export combined G-code.
5. Execute the combined G-code using UGS/GRBL on the 3-DOF writing robot.
