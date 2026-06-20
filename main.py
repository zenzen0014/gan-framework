from config import CONFIG
import evaluate
import train


def main() -> None:
    CONFIG.validate()
    train.main()
    evaluate.main()


if __name__ == "__main__":
    main()
