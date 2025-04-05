# Smooth Selection Plugin for GIMP 2.10.x

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GIMP Version](https://img.shields.io/badge/GIMP-2.10.x-green.svg)](https://www.gimp.org/)

Smooth Selection is a GIMP plugin that allows you to refine and smooth the edges of a selection using various smoothing algorithms. This plugin is particularly useful for cleaning up jagged or noisy selections, making them more visually appealing.

## Installation

1. **Download the Plugin**  
   Download the `smooth_selection.py` file from this repository.

2. **Locate Your GIMP Plugins Folder**  
   - On Windows:  
     `C:\Users\<YourUsername>\AppData\Roaming\GIMP\2.10\plug-ins`
   - On macOS:  
     `~/Library/Application Support/GIMP/2.10/plug-ins`
   - On Linux:  
     `~/.config/GIMP/2.10/plug-ins`

3. **Copy the Plugin**  
   Place the `smooth_selection.py` file into the `plug-ins` folder.

4. **Make the Plugin Executable**  
   Ensure the file is executable:
   - On Linux/macOS: Run `chmod +x smooth_selection.py` in the terminal.
   - On Windows: No additional steps are needed.

5. **Restart GIMP**  
   Restart GIMP to load the plugin.

## Usage

1. **Select an Area**  
   Use any selection tool in GIMP to define the area you'd like to smooth.

2. **Run the Plugin**  
   Navigate to the menu:  
   **`Select > Smooth Selection...`**

3. **Configure Options**  
   - **Smoothing Iterations (Slider)**: The number of smoothing passes to apply.  
     Example:  
     - 1 pass at strength 1 will make minimal changes to the selection.  
     - 10 passes at strength 1 will result in a much smoother selection but may significantly alter the shape.
   - **Smoothing Strength (Slider)**: Controls the intensity of each smoothing pass.  
     Example:  
     - Strength 1 applies subtle smoothing.  
     - Strength 10 applies aggressive smoothing, which may distort fine details.

4. **Boolean Options**  
   - **Preserve Curves**: If enabled, the plugin will attempt to maintain the original curves of the selection.  
   - **Preserve Path**: If enabled, the smoothed path will remain as a vector path in the image for further editing.

5. **Choose a Smoothing Method**  
   The plugin offers several smoothing methods. The most capable modes are:
   - **Sanding**: Smooths only outward-pointing bumps, preserving detail elsewhere.
   - **Inward Pixel Radius**: Pulls in bumps while preserving inward details, ideal for high-detail smoothing.

6. **Apply and Review**  
   Click OK to apply the smoothing. The selection will be updated based on the chosen settings.

## Example Workflow

1. Select an area with jagged edges using the Lasso Tool.
2. Open **`Select > Smooth Selection...`**.
3. Choose the **Sanding** method.
4. Set **Smoothing Iterations** to `5` and **Smoothing Strength** to `3`.
5. Enable **Preserve Curves** and disable **Preserve Path**.
6. Click OK to apply the smoothing.

## License

This plugin is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the plugin.
