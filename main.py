from dataset import DataPipeline  # match your actual filename


def main():
    pipeline = DataPipeline()
    train_loader, val_loader, test_loader = pipeline.load_dataset()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch labels: {labels}")


if __name__ == "__main__":
    main()