from flask import Flask, jsonify, request
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import openai
import os

app = Flask(__name__)

# Set the OpenAI API endpoint and your API key
API_ENDPOINT = "https://api.openai.com/v1/images/generations"
API_KEY = "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"


@app.route('/outpaint-image', methods=['GET'])
def outpaint_image():
    # Get the user's text input from the request
    user_input = request.args.get('text')
    response = openai.Image.create(
        prompt=user_input,
        n=1,
        size="1024x1024"
        )
    image_url = response['data'][0]['url']
    # Download the image using Pillow and convert to a numpy array
    image_data = requests.get(image_url).content
    image = Image.open(BytesIO(image_data))
    input_image = np.array(image)

    open("input_generated.png", "wb").write(image_data)

    input_image = cv2.imread("input_generated.png")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2RGBA)
    input_image = cv2.resize(input_image, (1024, 1024),
              interpolation = cv2.INTER_LINEAR)
    
    chunk_width = int(input_image.shape[1]/3)
    # Storing the first chunk
    first_chunk = input_image[:,0:chunk_width:,]

    # Second and third chunk
    partial_image = input_image[:,chunk_width:1024,]

    # Adding mask as new chunk and merging into 2nd and 3rd chunk and saving it
    last_chunk_image = np.zeros((1024,chunk_width,4))
    out_image_new = np.concatenate((partial_image,last_chunk_image),axis=1)
    out_image_new[chunk_width:,:3] = 0
    cv2.imwrite('partial.png',out_image_new)

    # Sending it to OPEN AI to fill mask image

    response = openai.Image.create_edit(
    image=open("partial.png", "rb"),
    mask=open("partial.png", "rb"),
    prompt=user_input,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']

    image_data = requests.get(image_url).content

    open("1.png", "wb").write(image_data)
    second_part = cv2.imread('1.png')

    # Merging first chunk to output image

    second_part = cv2.cvtColor(second_part, cv2.COLOR_RGB2RGBA)
    second_part = cv2.resize(second_part, (1024, 1024),
                interpolation = cv2.INTER_LINEAR)
    #final_image = np.concatenate((first_chunk,second_part),axis=1)
    final_image = np.hstack((first_chunk,second_part))
    cv2.imwrite('outpaint1.png',final_image)

    ### Staring Second Transformation


    final_image = cv2.imread('outpaint1.png')
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2RGBA)
    final_image_chunk_size = int(final_image.shape[1]/2)

    # Storing the first chunk
    pre_chunk = final_image[:,0:final_image_chunk_size:,]

    # Second chunk
    post_chunk = final_image[:,final_image_chunk_size:final_image.shape[1],]
    last_chunk_image = np.zeros((1024,1024-final_image_chunk_size-1,4))
    out_image_new = np.concatenate((post_chunk,last_chunk_image),axis=1)
    out_image_new[final_image_chunk_size:,:3] = 0
    cv2.imwrite('partial.png',out_image_new)


    response = openai.Image.create_edit(
    image=open("partial.png", "rb"),
    mask=open("partial.png", "rb"),
    prompt=user_input,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']

    # Downloading image
    image_req = requests.get(image_url)

    open("2.png", "wb").write(image_req.content)
    # Merging first chunk to output image
    second_part = cv2.imread('2.png')

    second_part = cv2.cvtColor(second_part, cv2.COLOR_RGB2RGBA)
    final_image = np.concatenate((pre_chunk,second_part),axis=1)
    cv2.imwrite('outpaint2.png',final_image)




    ### Staring Thid Transformation


    final_image = cv2.imread('outpaint2.png')
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2RGBA)
    final_image_chunk_size = int(final_image.shape[1]/2)

    # Storing the first chunk
    pre_chunk = final_image[:,0:final_image_chunk_size:,]

    # Second chunk
    post_chunk = final_image[:,final_image_chunk_size:final_image.shape[1],]
    last_chunk_image = np.zeros((1024,1024-final_image_chunk_size,4))
    out_image_new = np.concatenate((post_chunk,last_chunk_image),axis=1)
    out_image_new[final_image_chunk_size:,:3] = 0
    cv2.imwrite('partial.png',out_image_new)


    response = openai.Image.create_edit(
    image=open("partial.png", "rb"),
    mask=open("partial.png", "rb"),
    prompt=user_input,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']

    # Downloading image
    image_req = requests.get(image_url)

    open("3.png", "wb").write(image_req.content)
    # Merging first chunk to output image
    second_part = cv2.imread('3.png')
    second_part = cv2.cvtColor(second_part, cv2.COLOR_RGB2RGBA)
    final_image = np.concatenate((pre_chunk,second_part),axis=1)
    cv2.imwrite('output.png',final_image)


    # Convert the Pillow Image object to bytes
    combined_image = Image.fromarray(final_image)
    img_bytes = BytesIO()
    combined_image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
            
    # Return the combined image as a response
    return jsonify({'image': str(img_bytes)})

@app.route('/generate-image', methods=['POST'])
def generate_image():
    # Get the user's text input from the request
    user_input = request.json.get('text')

    # Set the text prompt and model parameters
    prompt = f"3D model of a equirectangular panorama. {user_input}"
    model = "image-alpha-001"
    size = "1024x1024"

    # Send the API request to generate the image
    response = requests.post(
        API_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        json={
            "model": model,
            "prompt": prompt,
            "num_images": 1,
            "size": size,
            "response_format": "url",
        },
    )

    # Check if the request was successful
    if response.status_code == 200:
        # Get the URL of the generated image from the response
        image_url = response.json()["data"][0]["url"]

        # Download the image using Pillow and convert to a numpy array
        image_data = requests.get(image_url).content
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)


        # Set the parameters for inpainting
        window_size = 15
        search_range = 30

        # Inpaint the left side of the image
        left = image_array[:, :512, :]
        left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        left_mask = np.zeros(left_gray.shape, dtype=np.uint8)
        left_mask[left_gray == 0] = 255
        left_inpaint = cv2.inpaint(left, left_mask, window_size, cv2.INPAINT_TELEA)

        # Inpaint the right side of the image
        right = image_array[:, 512:, :]
        right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        right_mask = np.zeros(right_gray.shape, dtype=np.uint8)
        right_mask[right_gray == 0] = 255
        right_inpaint = cv2.inpaint(right, right_mask, window_size, cv2.INPAINT_TELEA)

        # Combine the left and right sides of the image
        combined_inpaint = np.concatenate((left_inpaint, right_inpaint), axis=1)

        # Convert the combined image to a Pillow Image object
        combined_image = Image.fromarray(combined_inpaint)

        # Convert the Pillow Image object to bytes
        img_bytes = BytesIO()
        combined_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        # Display the combined image using Pillow
        combined_image = Image.fromarray(combined_inpaint)
        combined_image.show()
        temp = user_input.replace(" ","")
        # Set the filename based on the user input
        filename = f"combined_image-{temp}.png"

        # Save the combined image to a file using Pillow
        combined_image.save(filename)

               
        # Return the combined image as a response
        return jsonify({'image': str(img_bytes)})

    else:
        # Return the error message if the request failed
        return jsonify({'error': response.text})

    res = requests.get(f"http://localhost:3000/upload?file={filename}")
    if res.status_code == 200:
        return res.text


if __name__ == '__main__':
    app.run(debug=True)
