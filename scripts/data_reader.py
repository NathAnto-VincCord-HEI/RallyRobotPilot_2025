

if __name__ == "__main__":
    import numpy
    import pickle
    import lzma


    with lzma.open("record_2.npz", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshots")
        print(data[0].image)
        # print([e.car_speed for e in data])
        # print([e.raycast_distances for e in data])
