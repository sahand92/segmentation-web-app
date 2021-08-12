import io
import torch
from flask import Flask, jsonify, request, render_template, send_file, url_for
from PIL import Image
import numpy as np
import torch.nn.functional as F
from model.unet_model import UNet
from base64 import b64encode

app = Flask(__name__)

# import NN model here
with torch.no_grad():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=4)
    model_path = './trained_model/net_lentil_12082021_epoch150.pth'
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, device))
    net.eval()

# define image input transformation
def preprocess_img(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans

def preprocess_og_img(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans

def crop_center(pil_img, tile_size):
    w, h = pil_img.width, pil_img.height
    new_w = w//tile_size*tile_size
    new_h = h//tile_size*tile_size
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = (w + new_w) // 2
    bottom = (h + new_h) // 2
    center_cropped = pil_img.crop((left, top, right, bottom))
    return center_cropped

# create pil patches --> predict --> tile
def predict_tile(pil_img, tile_size):
    pil_img = crop_center(pil_img, tile_size)
    w, h = pil_img.width, pil_img.height
    full_mask = Image.new('RGB', (w, h))
    counts = np.zeros(4)
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            box = (j, i, j+tile_size, i+tile_size)
            cropped = pil_img.crop(box)
            # predict cropped
            cropped_mask = predict_img(cropped)
            counts_crop = pixel_percentage(cropped_mask)
            counts += counts_crop
            colormap = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
            cropped_mask = labeloverlay(cropped_mask, colormap)
            cropped_mask = np.array(cropped_mask * 255, dtype='uint8')
            cropped_mask = Image.fromarray(cropped_mask)

            full_mask.paste(cropped_mask, (j, i))
    number_of_tiles = (w//tile_size)*(h//tile_size)
    counts = counts/number_of_tiles
    return full_mask, counts

# predict image mask
def predict_img(pil_img):
    with torch.no_grad():
        img_tensor = torch.from_numpy(preprocess_img(pil_img, 1))
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        output = net(img_tensor)
    probs = F.softmax(output, dim=1)
    probs = probs.squeeze(0)
    probs = probs.detach().cpu().numpy()
    full_mask = np.moveaxis(probs, 0, -1)
    full_mask = full_mask > 0.5
    return full_mask

# count % pixels for each class
def pixel_percentage(predicted_mask):
    classes = predicted_mask.shape[2] if len(predicted_mask.shape) > 2 else 1
    counts = np.zeros(classes)
    for i in range(classes):
        counts[i-1] = np.sum(predicted_mask[:, :, i-1])
    counts = counts/(predicted_mask.shape[0]*predicted_mask.shape[1])*100
    return counts

# map mask channels to RGB colormap
def labeloverlay(mask, colormap):
    mask_colored = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3), dtype=float)
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    for i in range(classes):
        mask_nd = mask[:, :, i-1] * 1
        mask_nd = np.repeat(mask_nd[:, :, np.newaxis], 3, axis=2)
        mask_nd = mask_nd*colormap[i-1, :]
        mask_colored += mask_nd

    return mask_colored


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        bytes_img = file.read()
        pil_img = Image.open(io.BytesIO(bytes_img))
        #full_mask = predict_img(pil_img)
        tile_size = 200
        full_mask, counts = predict_tile(pil_img, tile_size)
        #predicted_mask = full_mask
        #colormap = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        #full_mask = labeloverlay(full_mask, colormap)
        #full_mask = np.array(full_mask*255, dtype='uint8')
        #full_mask = Image.fromarray(full_mask)
        pil_img = Image.fromarray(np.array(preprocess_og_img(pil_img, 1)*255, dtype='uint8'))
        pil_img = crop_center(pil_img, tile_size)
        # overlay full_mask and pil_img
        overlayed_img = Image.blend(pil_img.convert('RGBA'), full_mask.convert('RGBA'), .7)

        # create file object in memory
        file_object = io.BytesIO()

        #write mask to file object
        overlayed_img.save(file_object, 'PNG')

        # encode img to base64
        encoded_img = b64encode(file_object.getvalue())
        decoded_img = encoded_img.decode('utf-8')
        img_data = f"data:image/jpeg;base64,{decoded_img}"

        file_object.seek(0)

        #return render_template('result.html', image_path=img_data, name=pixel_percentage(predicted_mask))
        return render_template('result.html', image_path=img_data, name=counts)
        #return send_file(file_object, mimetype='image/PNG')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)