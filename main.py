from kivy.clock import Clock, mainthread
from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivymd.uix.snackbar import Snackbar
from kivy.uix.floatlayout import FloatLayout

from plyer import filechooser
import os
from transformers import pipeline
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
import pyperclip
import whisper

executor = ThreadPoolExecutor(max_workers=10)
Window.size = (360, 640)

class WhisperModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None  
        self.lock = threading.Lock()  
        self.thread = threading.Thread(target=self._load_model)
        self.thread.start() 

    def _load_model(self):
        with self.lock:
            print("Loading Whisper model...")
            self.model = whisper.load_model("tiny", device=self.device)  
            print("Whisper model loaded successfully!")

    def get_model(self):
        with self.lock:
            return self.model
        
class BARTModelLoader:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.summarizer = None  
        self.lock = threading.Lock() 
        self.thread = threading.Thread(target=self._load_model)
        self.thread.start()  

    def _load_model(self):
        with self.lock:
            print("Loading BART summarization model...")
            self.summarizer = pipeline(
                "summarization", model="facebook/bart-large-cnn", device=self.device
            )
            print("BART summarization model loaded successfully!")

    def get_summarizer(self):
        with self.lock:
            if self.summarizer is None:
                print("BART model is still loading...")
            return self.summarizer
        
whisper_loader = WhisperModelLoader()
bart_loader = BARTModelLoader()

class TranscribeWindow(Screen):
    def download_audio(self, vid_url, output_file='audio.mp3'):
        import yt_dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_file,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'keepvideo': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([vid_url])
        print(f"Downloaded audio as {output_file}")
    
    def convert_to_wav(self, input_file, output='audio.wav'):
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_file)
        audio.export(output, format='wav')
        print(f"Converted {input_file} to {output}")
    
    def clear_path(self):
        project_directory = os.path.dirname(os.path.abspath(__file__))
        audios = [
            project_directory+'\\audio.mp3', 
            project_directory+'\\audio.mp3.mp3', 
            project_directory+'\\audio.wav'
        ]
        for audio in audios:
            if os.path.exists(audio):
                os.remove(audio)
            else:
                print("Audio not found")

    def transcribe_audio(self, audio_file):
        model = whisper_loader.get_model()  
        if not model:
            return "Model is still loading. Please try again shortly."
        result = model.transcribe(audio_file)
        return result['text']
    
    def transcribe(self):
        if not whisper_loader.get_model():
            self.ids.transcribed_text.text = "Loading model... Please wait."
            return
        self.clear_transcribed_text()
        self.ids.progress_bar.value = 0
        if self.ids.url.text:
            print("Threading starting...")
            video = self.ids.url.text
            thread = threading.Thread(target=self.run_transcription, args=(video,))
            thread.daemon = True
            thread.start()
            print("Threading started")
        elif self.ids.file_path.text:
            video = self.ids.file_path.text
            thread = threading.Thread(target=self.run_transcription, args=(video, False))
            thread.daemon = True
            thread.start()
        else:
            return f"No file found"
        
    def run_transcription(self, video, is_url = True):
        try:
            print("Running transcription...")
            self.start_progress()
            self.ids.transcribed_text.readonly = True
            if is_url:
                self.download_audio(video)
                self.convert_to_wav('audio.mp3')
                transcription = self.transcribe_audio('audio.wav')
                self.clear_path()
            else:
                transcription = self.transcribe_audio(video)
                self.clear_file_path()
        except Exception as e:
            transcription = f"Error: {str(e)}"
        self.update_transcribed_text(transcription)
        self.ids.transcribed_text.readonly = False

    @mainthread
    def update_transcribed_text(self, transcipt):
        self.ids.transcribed_text.text = transcipt
    @mainthread
    def clear_file_path(self):
        self.ids.file_path.text = ""
    def clear_transcribed_text(self):
        self.ids.transcribed_text.text = ''
    def file_chooser(self):
        self.ids.file_path.text = ''
        extensions = [
            '.mp4',
            '.avi',
            '.mov',
            '.wmv',
            '.flv',
            '.mkv',
            '.webm',
            '.mpeg',
            '.mpg',
            '.3gp',
            '.m4v',
            '.ts',
            '.f4v'
        ]
        vid = filechooser.open_file() #file is stored in a list
        if vid: #checks if vid variable is empty or not
            path = vid[0]
            if any(path.lower().endswith(extension) for extension in extensions):
                self.ids.file_path.text = path
            else:
                self.ids.file_path.text = "Invalid File Format"
        else:
            print("No file")
        
    def get_transcribed_text(self):
        """Return the text from the transcribed_text field."""
        return self.ids.transcribed_text.text
    
    def copy_text(self):
        pyperclip.copy(self.ids.transcribed_text.text)
        pyperclip.paste()
        Snackbar(text="Text copied!").show()

    def start_progress(self):
        Clock.schedule_interval(self.update_progress, 0.5)
    
    def update_progress(self, dt):
        if self.ids.progress_bar.value < 90:
            self.ids.progress_bar.value += 10
        elif self.ids.transcribed_text.text:
            self.ids.progress_bar.value += 10
            Clock.unschedule(self.update_progress)
            self.ids.progress_bar.value = 0

class SummaryWindow(Screen):
    def toggle_switches(self, instance, value):
        if not self.ids.par.active and not self.ids.bulleted.active:
            instance.active = True 
        elif instance == self.ids.par and value:
            self.ids.bulleted.active = False
        elif instance == self.ids.bulleted and value:
            self.ids.par.active = False
            
    def chunk_text(self, text, chunk_size=500):
        print("Chunking text...")
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    def get_length(self):
        if self.ids.length.value == 1:
            return 50, 10
        elif self.ids.length.value == 2:
            return 80, 50
        elif self.ids.length.value == 3:
            return 100, 80
    
    def summarize(self):
        self.clear_summarized_text()
        self.ids.summarized_text.readonly = True
        thread = threading.Thread(target=self.run_summarization)
        thread.start()
    
    def run_summarization(self):
        summarizer = bart_loader.get_summarizer()
        transcribed_text = self.manager.get_screen('transcribe').get_transcribed_text()
        text_chunks = self.chunk_text(transcribed_text)
        max, min = self.get_length()
        if summarizer:
            self.start_progress()
            if self.ids.par.active:
                summary = [summarizer(chunk, max_length=max, min_length=min, do_sample=False)[0]['summary_text'] 
                            for chunk in text_chunks]
                self.update_summarized_text(summary[0])
            else:
                summary = [summarizer(chunk, max_length=80, min_length=50, do_sample=False)[0]['summary_text'] 
                            for chunk in text_chunks]
                note_style = "Summary:\n"+"\n".join([f"• {line.strip()}" for s in summary for line in s.split('. ') if line])
                self.update_summarized_text(note_style)
        else:
            print("Please wait, the model is still loading.")
    @mainthread
    def update_summarized_text(self, text):
        self.ids.summarized_text.text = text
    def clear_summarized_text(self):
        self.ids.summarized_text.text = ''
    def copy_text(self):
        pyperclip.copy(self.ids.summarized_text.text)
        pyperclip.paste()

    def start_progress(self):
        Clock.schedule_interval(self.update_progress, 0.5)
    
    def update_progress(self, dt):
        if self.ids.progress_bar.value < 90:
            self.ids.progress_bar.value += 10
        elif self.ids.summarized_text.text:
            self.ids.progress_bar.value += 10
            Clock.unschedule(self.update_progress)
            self.ids.progress_bar.value = 0

class WindowManager(ScreenManager):
    pass
class main(MDApp):
    def build(self):
        
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "BlueGray"
        self.theme_cls.primary_hue = "A700"
        kv_path = os.path.join(os.path.dirname(__file__), "styles.kv")
        kv = Builder.load_file(kv_path)
        return kv
    
    def switch_to_summary(self):
        self.root.transition = FadeTransition(duration=0.5)
        self.root.current = 'summary'

    def switch_to_transcribe(self):   
        self.root.transition = FadeTransition(duration=0.5)
        self.root.current = 'transcribe'

    dialog = None
    def show_alert_dialog(self):
        if not self.dialog:
            # Create content for the popup
            content = FloatLayout()
            
            # Create buttons
            cancel_button = Button(
                text="CANCEL",
                size_hint=(0.4, 0.2),
                pos_hint={'x': 0.05, 'y': 0.1}
            )
            cancel_button.bind(on_release=self.dismiss_dialog)
            
            discard_button = Button(
                text="DELETE",
                size_hint=(0.4, 0.2),
                pos_hint={'x': 0.55, 'y': 0.1}
            )
            discard_button.bind(on_release=self.clear_text_and_dismiss)
            
            # Add buttons and label to the layout
            content.add_widget(Label(
                text="Do you want to clear the text field?",
                pos_hint={'center_x': 0.5, 'center_y': 0.7}
            ))
            content.add_widget(cancel_button)
            content.add_widget(discard_button)

            # Create popup dialog
            self.dialog = Popup(
                title="Alert",
                content=content,
                size_hint=(0.8, 0.4),
                auto_dismiss=False,
                background_color=(0, 0, 1, 1)
            )
        
        # Open the dialog
        self.dialog.open()

    def dismiss_dialog(self, instance):
        self.dialog.dismiss()
    def clear_text_and_dismiss(self, instance):
        current_screen = self.root.current
        if current_screen == 'transcribe':
            self.root.get_screen('transcribe').clear_transcribed_text()
        elif current_screen == 'summary':
            self.root.get_screen('summary').clear_summarized_text()
        self.dismiss_dialog(instance)
if __name__=="__main__":
    main().run()