from src.data_engineering.eia.pipeline import EIAPipeline


if __name__ == "__main__":
    pipeline = EIAPipeline()

    result = pipeline.run("commercial_crude_oil")

    print("\nEIA INVENTORY PIPELINE RESULTS")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")
