import torch


device = None


def set_device(args):
    global device

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu_id))
    else:
        device = torch.device('cpu')

CONCEPTS = [
    'I', 'you', 'we', 'this', 'that', 'who', 'what', 'not', 'all',
    'many', 'one', 'two', 'big', 'long', 'small', 'woman', 'man',
    'person', 'fish', 'bird', 'dog', 'louse', 'tree', 'seed',
    'leaf', 'root', 'bark', 'skin', 'flesh', 'blood', 'bone',
    'grease', 'egg', 'horn', 'tail', 'feather', 'hair', 'head',
    'ear', 'eye', 'nose', 'mouth', 'tooth', 'tongue', 'claw',
    'foot', 'knee', 'hand', 'belly', 'neck', 'breast', 'heart',
    'liver', 'drink', 'eat', 'bite', 'see', 'hear', 'know', 'sleep',
    'die', 'kill', 'swim', 'fly', 'walk', 'come', 'lie',
    'sit', 'stand', 'give', 'say', 'sun', 'moon', 'star',
    'water', 'rain', 'stone', 'sand', 'earth', 'cloud',
    'smoke', 'fire', 'ash', 'burn', 'path', 'mountain',
    'red', 'green', 'yellow', 'white', 'black', 'night',
    'hot', 'cold', 'full', 'new', 'good', 'round', 'dry', 'name'
]
ASJP_EXT_COLUMNS = [
    'names', 'wls_fam', 'wls_gen', 'e', 'hh', 'lat', 'lon', 'pop', 'wcode', 'iso'
]
