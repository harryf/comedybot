# ComedyBot

ComedyBot is a tool designed to transcribe audio files and analyze them for laughter. It utilizes advanced GPT models to perform these tasks, providing a user-friendly interface for metadata input and outputting results in a specified directory.

## Requirements

- **OSX**: Minimum version Sonoma 14.6.1
- **Homebrew**: Recommended for package management. [Install Homebrew](https://brew.sh/)
- **Python**: Tested with Python 3.12
- **GitHub**: Ensure GitHub is installed and you have an active GitHub account

## Installation

1. **Open Terminal**: Launch the terminal application on your Mac.

2. **Clone the Project**:
   ```bash
   git clone git@github.com:harryf/comedybot.git
   ```

3. **Navigate to the Project Directory**:
   ```bash
   cd comedybot
   ```

4. **Create a Python Virtual Environment**:
   - Use the built-in Python 3.x method:
     ```bash
     python3 -m venv venv
     ```

5. **Activate the Virtual Environment**:
   - Run the following command:
     ```bash
     source venv/bin/activate
     ```

6. **Install the Required Packages**:
   - Use pip to install the dependencies listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

1. **Prepare Input and Output Folders**:
   - Create an `input` and an `output` folder, for example, on your Desktop.

2. **Add Audio Files**:
   - Copy your audio files (supported formats: .wav, .mp3, .m4a) into the `input` folder.

3. **Run the Transcription and Analysis**:
   - Ensure you are in the `comedybot` directory and the virtual environment is active.
   - Execute the following command, replacing `<yourname>` with your actual username:
     ```bash
     python ./comedy_set_analysis/audio_transcription_agent/audio_transcript_process.py -i /Users/<yourname>/Desktop/input -o /Users/<yourname>/Desktop/output
     ```

   - This process will download necessary GPT models, transcribe the audio, and analyze it for laughter. A GUI will prompt you to provide metadata, and the results will be saved in the `output` folder.

**Note**: The process may take a significant amount of time depending on the length of the audio file.

## Additional Information

- Ensure your system meets all the requirements before proceeding with the installation.
- The initial download of GPT models can be large, so ensure you have sufficient disk space and a stable internet connection.
