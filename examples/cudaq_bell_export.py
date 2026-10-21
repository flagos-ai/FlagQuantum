"""Export a Bell circuit with the optional Linux CUDA-Q SDK on qpp-cpu."""


def main() -> None:
    import cudaq
    import numpy as np

    import flagquantum as fq
    from flagquantum.ecosystem.cudaq import to_cudaq

    cudaq.set_target("qpp-cpu")
    kernel = to_cudaq(fq.Circuit(2).h(0).cx(0, 1))
    probabilities = np.abs(np.asarray(cudaq.get_state(kernel))) ** 2
    print("Bell probabilities: [" + ", ".join(f"{p:.1f}" for p in probabilities) + "]")


if __name__ == "__main__":
    main()
