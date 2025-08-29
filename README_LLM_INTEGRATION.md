# StreamDiffusion with LLM Integration

This enhanced version of the StreamDiffusion example integrates LLM capabilities to provide a more interactive and creative experience on macOS.

## Setup

1. Install the required dependencies:
```bash
pip install python-dotenv keyboard
```


## Features

- **Enhanced Prompts**: The system takes your basic prompts and enhances them with additional descriptive terms
- **Random Initial Prompts**: Creative starting prompts are generated automatically based on different themes
- **Smooth Transitions**: Previous images are displayed during prompt changes to avoid freezing
- **Interactive Controls**:
  - Press 'r' for a new random prompt
  - Press 's' to save the current image to gallery
  - Press 'p' to view prompt history
  - Press 'q' to quit
- **Image Gallery**: Images are saved with their prompts and timestamps in the 'gallery' folder
- **Text Overlay**: Current prompt is displayed on the generated image

## Running the Example

```bash
python -m app.examples.web_camera
```

You can also send prompts via a TCP connection to localhost:65432.
