# 📼 PyVidNote

**PyVidNote** is a mobile application built using **Python** that transcribes and summarizes video content. It utilizes **OpenAI's Whisper** model for transcription and **Facebook AI's BART** model for summarization, enabling users to comprehend and review video content efficiently by extracting key information.

## 🚀 Features

- 🎙️ **Transcribe Videos**: Automatically converts video audio to text using the Whisper model.
- 🧠 **Summarize Content**: Extracts the most important points from the transcription using BART summarization.
- 📱 **Mobile-Friendly UI**: Developed with Kivy and KivyMD for a responsive and visually appealing user interface.
- 📂 **File Selection**: Easily choose video files using a built-in file chooser.
- 📋 **Clipboard Support**: Copy transcripts or summaries to your clipboard instantly.

## 🛠️ Technologies Used

- [Python](https://www.python.org/)
- [Kivy](https://kivy.org/) & [KivyMD](https://github.com/kivymd/KivyMD) – for cross-platform mobile UI
- [Whisper](https://github.com/openai/whisper) – for speech-to-text transcription
- [BART (Facebook AI)](https://huggingface.co/facebook/bart-large-cnn) – for abstractive summarization
- [Transformers](https://huggingface.co/transformers/) – for accessing BART model
- [Plyer](https://github.com/kivy/plyer) – for native file access
- [Pyperclip](https://pypi.org/project/pyperclip/) – for clipboard support
- [Torch](https://pytorch.org/) – deep learning framework used by Whisper and BART

## 📦 Dependencies

Ensure the following Python packages are installed:

```bash
pip install kivy kivymd plyer pyperclip torch torchvision torchaudio \
            git+https://github.com/openai/whisper.git \
            transformers
```
## Contact
Email: catabayjosiah19 @gmail.com
