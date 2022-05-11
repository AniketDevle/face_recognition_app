import boto3
from face_recognizer import get_prediction
import base64
from aws_credentials import Access_key, Secret_access_key

def get_dynamo_result(prediction):
    client = boto3.Session(
        region_name='us-east-1',
        aws_access_key_id=Access_key,
        aws_secret_access_key=Secret_access_key,
    )

    dynamodb = client.resource('dynamodb')
    table = dynamodb.Table('student_info')
    response = table.get_item(
        Key={
            'name': prediction
        }
    )
    return response['Item']


def face_recognition_handler(event, context):
    #image_encoding = event
    image_encoding = event['image']
    decoded_image = base64.b64decode(image_encoding)
    prediction = get_prediction(decoded_image)
    return get_dynamo_result(prediction)

'''
if __name__ == '__main__':
    img_path = "20220422_151516.png"
    with open(img_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read())
        text_file = open("base64.txt", "wb")
        text_file.write(encoded_image)
        print(face_recognition_handler(encoded_image, "Test"))
        text_file.close()
'''