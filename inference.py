import yaml, torch
from gaitgl.model import Model
from gaitgl.inference_utils import embedding

def main():

    conf = yaml.safe_load(open("config.yaml"))
    ckpt = conf["checkpoint"]

    model = Model(
        conf["hidden_dim"],
        conf["margin"],
        (conf["batch_P"], conf["batch_K"])
    )

    state = torch.load(ckpt, map_location="cpu")
    model.encoder.load_state_dict(state)
    model = model.cuda()

    seq = input("Ruta de carpeta con PNGs: ").strip()
    emb = embedding(model, seq, conf["frame_num"], conf["img_size"])
    print("Vector embedding:", emb.shape)
    print(emb)

if __name__ == "__main__":
    main()
