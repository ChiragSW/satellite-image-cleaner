from pipeline import EnhancementPipeline

if __name__ == "__main__":
    pipeline = EnhancementPipeline()
    pipeline.run(
        image_path="testing-env/inputs/0.png",
        save_path="testing-env/outputs/satelliteimagetestingenhanced.png"
    )