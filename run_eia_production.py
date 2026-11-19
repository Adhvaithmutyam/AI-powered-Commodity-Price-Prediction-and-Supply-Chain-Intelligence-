from src.data_engineering.eia.production_pipeline import EIAProductionPipeline

if __name__ == "__main__":
    result = EIAProductionPipeline().run()
    for key, value in result.items():
        print(f"{key}: {value}")
