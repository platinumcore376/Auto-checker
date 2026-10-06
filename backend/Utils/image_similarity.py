from PIL import Image

_model = None
_preprocess = None
_tokenizer = None
_device = None

def _get_model():
    global _model, _preprocess, _tokenizer, _device
    if _model is None:
        import torch
        import open_clip
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model, _preprocess, _tokenizer = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        _model = _model.to(_device)
    return _model, _preprocess, _device

def transform(x):
    import torch
    return torch.where(
        x >= 0.9, x,  # Keep values ≥ 0.9 unchanged
        0.5 * (x / 0.85)  # Scale down everything below 0.9 significantly
    )



def image_similarity(truth_file, image_files):
    import torch
    model, preprocess, device = _get_model()

    # preprocessing the image
    truth_tensor = preprocess(truth_file).unsqueeze(0).to(device)
    
    # preprocessing the images
    test_tensors = torch.stack([preprocess(img) for img in image_files]).to(device)
    
    #generating the image embedding
    with torch.no_grad():
        truth_embed = model.encode_image(truth_tensor)
        test_embeds = model.encode_image(test_tensors)
        
    similarities = torch.nn.functional.cosine_similarity(truth_embed, test_embeds)
    
    return transform(similarities).cpu().numpy()

if __name__ == '__main__':
    print(image_similarity(Image.open("backend/Utils/truth.png"), ["backend/Utils/image.png", "backend/Utils/image1.png"]))
