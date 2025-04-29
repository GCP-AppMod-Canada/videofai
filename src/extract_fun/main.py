import functions_framework
from google.cloud import storage
from random import randint
import json
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """


    # Expected JSON structure
    #  {
    #     "prompt": <user_prompt>,
    #     "project": <project_name>,
    #     "location": GCP region,
    #     "model":  model selected by user,
    #     "video_input": original file,
    #     "path": "gs://<bucketname>/appsheet/data//"+path, 
    #     "bucket": GCS bucket name where the original file is stored
    #   }

    item = request.get_json(silent=True)

    # enrich the user prompt with our own prompt
    prompt = item["prompt"]+"""il faut etre très attentif au début et à la fin du clip afin de ne couper aucun intervenant au milieu d'une phrase. Retourner le résultat au format JSON uniquement avec les clés suivantes : "debut", "duree" avec une précision à la milliseconde.
voici un exemple de reponse {"debut": 00:01.345, "fin": 00:29.690}"""

    # Add a System Instructions to improve the result
    textsi = """you are a video editor specialist and you extract clips from videos provided to you. translate to english any prompt you receive"""
    

    # call Vertex AI and save result in response

    vertexai.init(project=item["project"], location=item["location"])

    video_input = Part.from_uri(
        mime_type= "video/mp4",
        #uri= "gs://"+item["path"]+item["video_input"],
        uri=f"gs://{item['bucket']}/{item['path']}{item['video_input']}"
    )

    
    print("file: ", item["video_input"])
    print("path: ", item["path"])
    print("model: ", item["model"])
    print("prompt; ", item["prompt"])
    print("bucket: ", item["bucket"])
    

    # Gen AI safety settings
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]

    # prompt = item["prompt"]
    #prompt = """ extract a 20 seconds clip featuring otters"""
    model = GenerativeModel(
        item["model"],
        system_instruction=[textsi]
    )

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
        "response_mime_type": "application/json",
        "response_schema": {"type":"OBJECT","properties":{"debut":{"type":"STRING"},"fin":{"type":"STRING"}}},
    }

    response = model.generate_content(
        [video_input, prompt],
        generation_config= generation_config,
        safety_settings= safety_settings,
        stream=False,
    )


    timestamp = json.loads(response.text)

    # now that we have debut and fin for the clip, we can start the extraction process

    print("debut: ",timestamp["debut"])
    print("fin: ",timestamp["fin"])
  
    # download original file from GCS

    # # Initialise a client
    storage_client = storage.Client(item["project"])
    # # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(item["bucket"])
    # # Create a blob object from the filepath
    #blob = bucket.blob(item["path"]+item["video_input"])

    blob_path = item["path"]+item["video_input"]
    blob = bucket.blob(blob_path)


    # Download the file to /tmp in the function memory
    blob.download_to_filename('/tmp/downloadedVideo.mp4')

    video_original = VideoFileClip('/tmp/downloadedVideo.mp4')

    # extract a clip from orignal video downloaded
    clip = video_original.subclip(timestamp['debut'], timestamp['fin'])

    clip.write_videofile("/tmp/clip.mp4", logger=None)

    # upload to GCS
    filename = 'clipextrait-'+str(randint(1, 10000))+'.mp4'

    # extracted clip will be saved in path/resultats
    blob = bucket.blob(f"{item["path"]}resultats/{filename}")

    blob.upload_from_filename('/tmp/clip.mp4')

    print(f"{item["path"]}resultats/{filename}")
    return(f"{item["path"]}resultats/{filename}")