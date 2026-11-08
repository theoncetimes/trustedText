"""Train the contrastive detector on data/ and save it to models/."""

import argparse

from trustedtext.pipeline import Config, TrustedText


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show UMAP/t-SNE plots of raw and projected embeddings (blocks until closed)",
    )
    args = parser.parse_args()

    detector = TrustedText(Config(epochs=args.epochs, hidden_dim=args.hidden_dim))
    detector.setup()
    detector.train()

    if args.visualize:
        detector.visualize(method="umap", use_projection=False)
        detector.visualize(method="umap", use_projection=True)

    metrics = detector.evaluate()
    detector.save()
    detector.save_stats(test_metrics=metrics)


if __name__ == "__main__":
    main()
