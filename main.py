# Importamos las dependencias
import gradio as gr
import whisper 
from translate import Translator
from elevenlabs import ElevenLabs, VoiceSettings

# Definimos el API key de ElevenLabs
ELEVENLABS_API_KEY = "skhdkashdkashdkahskdkaskhd"

# Función principal para procesar el audio
def traductor(archivo_de_audio):
    try:
        # Cargamos el modelo de Whisper
        modelo = whisper.load_model("base")
        # Transcribimos el audio a texto
        resultado = modelo.transcribe(archivo_de_audio, language="Spanish", fp16=False)
        transcripcion = resultado["text"]
        print(f"Texto original: {transcripcion}")
    except Exception as e:
        raise gr.Error(f"Se ha producido un error procesando el audio: {str(e)}")  
    
    try:
        # Traducimos el texto
        en_traduccion = Translator(from_lang="es", to_lang="en").translate(transcripcion)
        print(f"Texto traducido: {en_traduccion}")
    except Exception as e:
        raise gr.Error(f"Se ha producido un error traduciendo el texto: {str(e)}") 
    
    try:
        # Convertimos el texto traducido a audio
        cliente = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        respuesta = cliente.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB", 
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_traduccion,
            model_id="eleven_turbo_v2", 
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            )
        )
        
        # Guardamos el audio en una ruta específica
        ubicacion_del_path = r"C:\Users\Julian\Desktop\Python\Traductor_con_IA\audio_en\en.mp3"
        with open(ubicacion_del_path, "wb") as f:
            for chunk in respuesta:
                if chunk:
                    f.write(chunk)
    except Exception as e:
        raise gr.Error(f"Se ha producido un error creando el audio: {str(e)}") 
    
    return ubicacion_del_path

# Interfaz web con Gradio
web = gr.Interface(
    fn=traductor,
    inputs=gr.Audio(type="filepath", label="Español"),
    outputs=gr.Audio(label="Inglés"),
    title="Traductor de voz con IA",
    description="Traductor de voz con inteligencia artificial en ingles."
)

web.launch()
