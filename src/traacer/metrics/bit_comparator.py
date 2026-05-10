from queue import Queue

bit_queue = Queue()

def compare_bits():
    original_bits = list(bit_queue.get())
    received_bits = list(bit_queue.get())

    if len(original_bits) != len(received_bits):
        print(f"Length mismatch: original={len(original_bits)}, received={len(received_bits)}")

    n = min(len(original_bits), len(received_bits))
    errors = [i for i in range(n) if original_bits[i] != received_bits[i]]

    print("Original: ", "".join(map(str, original_bits)))
    print("Received: ", "".join(map(str, received_bits)))

    if errors:
        markers = ["^" if i in errors else " " for i in range(n)]
        print("Errors:   ", "".join(markers))
        print(f"\nBit errors: {len(errors)} / {n} ({len(errors) / n:.2%})")
        print("Error indices:", errors)
    else:
        print("\nNo bit errors.")

    if len(original_bits) != len(received_bits):
        extra_original = original_bits[n:]
        extra_received = received_bits[n:]

        if extra_original:
            print("Extra original bits:", "".join(map(str, extra_original)))
        if extra_received:
            print("Extra received bits:", "".join(map(str, extra_received)))
