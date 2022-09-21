top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
    for image_index in range(len(images)):
        print("Image #{}".format(image_index))
        for class_id, name, y_proba in top_K[image_index]:
            print(" {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
        print()
  
