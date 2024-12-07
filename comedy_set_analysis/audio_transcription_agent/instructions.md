# Agent Role

The Audio Transcription Agent is responsible for transcribing comedy set audio files and detecting audience reactions using OpenAI's Whisper model and the fork Whisper-at for sound detection. It uses multiple tools to produce a final output JSON file that contains the transcript and the audience reactions (including their score) ordered by the time each event happens.

# Goals

- Transcribe audio files accurately.
- Detect sounds in the audio files accurately.
- Clean up the raw JSON from the transcript and the sounds.

# Process Workflow

1. **Receive a directory path containing audio files.**
   - The process starts by receiving the input directory path where the audio files are stored.
   
2. **Find all audio files in the directory using `audio_file_finder_tool`.**
   - This tool scans the directory and lists all the audio files available for processing.
   - If no audio files are found, the process logs this information and terminates early.

3. **Process each audio file sequentially:**
   a. **Prepare the audio file using `audio_preparation_tool`.**
      - This tool prepares the audio file for transcription and sound detection by performing necessary pre-processing steps.
      - It also creates a new subdirectory for each audio file to keep processed data organized. It makes a copy of the original file in the input directory to the new subdirectory. The subdirectory is named using the date, title, and location of the set. It will also be used by later tools to store metadata, cleaned transcript, and cleaned sounds.
      - If the audio file has already been prepared (detected by checking if the new subdirectory already exists), the file is skipped, and the process moves to the next audio file.
      - It also provides a rollback method that removes the subdirectory and it's contents, if any errors occur later in the process.
      
   b. **Capture metadata using `metadata_capture_tool`.**
      - This tool captures initial metadata about the audio file and stores it in the new subdirectory. It does so by prompting the user to input the title, date, and location of the set. Note that this works on MacOS only.
      - If metadata capture fails (i.e. user hits cancel), the process logs an error, performs a rollback of the created subdirectory (using the rollback method in `audio_preparation_tool`), and terminates the processing of the current audio file.
      - Note that effectively processing of further audio files pauses until the metadata is captured from the user.
      
   c. **Transcribe the audio using `audio_transcription_tool`.**
      - This tool uses transcription models to convert speech in the audio file to text, saving the output in the new subdirectory. It uses the OpenAI Whisper model. Note that this step can be slow for long audio files.
      
   d. **Detect sounds using `sound_detection_tool`.**
      - This tool identifies specific sounds (like audience reactions) within the audio file and logs them separately. It does so using the [whisper-at model](https://github.com/openai/whisper-at) which is a fork of the OpenAI Whisper model that is specifically trained to detect sound events in audio. Note that this step can also be slow for long audio files.
      
   e. **Clean the raw transcript file using `transcript_analyser_tool`.**
      - This tool processes the raw transcript to remove errors and irrelevant data, producing a cleaned transcript.
      
   f. **Clean the raw sounds file using `sound_analyser_tool`.**
      - Similar to the transcript analyser, this tool cleans the raw sound detection data, adding the laughter score to each sound event then calculating the laughs per minute.
      
   g. **Update the metadata file using `metadata_update_tool`.**
      - After processing is complete, this tool updates the metadata with new information obtained from the transcription and sound analysis namely the laughs per minute and total set length (time in seconds).

This detailed workflow ensures that each part of the audio file processing is handled systematically, with checks and balances to manage errors and ensure data integrity. In future this could be re-written to handle some steps of the process in parallel.