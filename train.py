import yaml, torch, matplotlib.pyplot as plt
from pathlib import Path
from gaitgl.data_loader import load_data
from gaitgl.model import Model

def plot_curve(values, title, savepath):
    plt.figure(figsize=(6,4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.savefig(savepath)
    plt.close()

def main():

    conf = yaml.safe_load(open("config.yaml","r"))
    work = Path(conf["work_dir"])
    plots = Path(conf["plots_dir"])
    work.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    train_src, test_src = load_data(
        conf["dataset_preprocessed"],
        conf["resolution"],
        conf["dataset"],
        conf["pid_num"],
        conf["pid_shuffle"]
    )

    P = conf["batch_P"]; K = conf["batch_K"]
    model = Model(
        conf["hidden_dim"],
        conf["margin"],
        (P,K)
    ).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=conf["lr"])

    loss_curve = []
    for it in range(conf["total_iter"]):

        batch, label = train_src.next_batch(P,K)
        emb = model(batch.cuda())
        loss = model.triplet_loss(emb, label.cuda())

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_curve.append(loss.item())

        if it % 1000 == 0:
            print(f"[{it}] loss={loss.item():.4f}")

    torch.save(model.encoder.state_dict(), work/"encoder.ptm")

    plot_curve(loss_curve, "Training Loss", plots/"loss.png")

if __name__ == "__main__":
    main()
