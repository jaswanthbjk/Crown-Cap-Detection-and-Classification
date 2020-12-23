import argparse
import io
import os
from collections import namedtuple

import pandas as pd
import tensorflow as tf
from PIL import Image
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
from six import string_types


def load_labelmap(path):
    """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
    with tf.io.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    return label_map


def get_label_map_dict(label_map_path_or_proto,
                       use_display_name=False,
                       fill_in_gaps_and_background=False):
    """Reads a label map and returns a dictionary of label names to id.

  Args:
    label_map_path_or_proto: path to StringIntLabelMap proto text file or the
      proto itself.
    use_display_name: whether to use the label map items' display names as keys.
    fill_in_gaps_and_background: whether to fill in gaps and background with
    respect to the id field in the proto. The id: 0 is reserved for the
    'background' class and will be added if it is missing. All other missing
    ids in range(1, max(id)) will be added with a dummy class name
    ("class_<id>") if they are missing.

  Returns:
    A dictionary mapping label names to id.

  Raises:
    ValueError: if fill_in_gaps_and_background and label_map has non-integer or
    negative values.
  """
    if isinstance(label_map_path_or_proto, string_types):
        label_map = load_labelmap(label_map_path_or_proto)
    else:
        label_map = label_map_path_or_proto

    label_map_dict = {}
    for item in label_map.item:
        if use_display_name:
            label_map_dict[item.display_name] = item.id
        else:
            label_map_dict[item.name] = item.id

    if fill_in_gaps_and_background:
        values = set(label_map_dict.values())

        if 0 not in values:
            label_map_dict['background'] = 0
        if not all(isinstance(value, int) for value in values):
            raise ValueError('The values in label map must be integers in order to'
                             'fill_in_gaps_and_background.')
        if not all(value >= 0 for value in values):
            raise ValueError('The values in the label map must be positive.')

        if len(values) != max(values) + 1:
            # there are gaps in the labels, fill in gaps.
            for value in range(1, max(values)):
                if value not in values:
                    label_map_dict[str(value)] = value

    return label_map_dict


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def create_tf_example(group, path, category_idx):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(float(row['xmin']) / width)
        xmaxs.append(float(row['xmax']) / width)
        ymins.append(float(row['ymin']) / height)
        ymaxs.append(float(row['ymax']) / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(category_idx[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating tfrecords from images and csv file")
    parser.add_argument("--path_to_images", type=str, help="folder that contains images",
                        default="data/train/images")
    parser.add_argument("--path_to_annot", type=str, help="full path to annotations csv file",
                        default="annotations.csv")
    parser.add_argument("--path_to_label_map", type=str, help="full path to label_map file",
                        default="label_map.pbtxt")
    parser.add_argument("--path_to_save_tfrecords", type=str, help="This path is for saving the generated tfrecords",
                        default="data/myrecord.record")
    args = parser.parse_args()

    csv_path = args.path_to_annot
    images_path = args.path_to_images
    print("images path : ", images_path)
    print("csv path : ", csv_path)
    print("path to output tfrecords : ", args.path_to_save_tfrecords)
    label_map_dict = get_label_map_dict(args.path_to_label_map)
    writer = tf.io.TFRecordWriter(args.path_to_save_tfrecords)

    examples = pd.read_csv(csv_path)
    print("Generating tfrecord .... ")
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_path, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(args.path_to_save_tfrecords))
