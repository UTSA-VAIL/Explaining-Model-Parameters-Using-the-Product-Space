from genericpath import isdir
import os
import shutil

if __name__ == '__main__':
    imagenet_directory = '/data/progressive_data_dropout/imagenet/our_train'
    val_directory = '/data/progressive_data_dropout/imagenet/our_val'
    name_paths = os.listdir(imagenet_directory)
    
    for path in name_paths:
        full_path = os.path.join(imagenet_directory, path)

        if os.path.isdir(full_path):
            image_names = sorted(os.listdir(full_path))
            print(len(image_names))

            # Make directory in validation
            class_validation_dir = os.path.join(val_directory, path)
            if not os.path.exists(class_validation_dir):
                os.mkdir(class_validation_dir)

            # Get last 5 percent from the list
            total_examples = int(len(image_names) * 0.05)
            # print('Total examples', total_examples)
            start_index = len(image_names) - (total_examples)
            # print('Start index', start_index)
            validation_files = image_names[start_index:]

            # print('Validation len', len(validation_files))

            for val_file in validation_files:
                old_path = os.path.join(imagenet_directory, path, val_file)
                new_path = os.path.join(class_validation_dir, val_file)

                # print(old_path)
                # print(new_path)
                shutil.move(old_path, new_path)

        else:
            continue