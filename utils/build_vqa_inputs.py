import numpy as np
import json
import os
import argparse
from text_helper import tokenize, VocabDict
from collections import Counter


def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, image_set):
    print('building vqa %s dataset' % image_set)
    if image_set in ['train2015', 'val2015']:
        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)['annotations']
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
    else:
        load_answer = False
    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    coco_set_name = image_set.replace('-dev', '')
    abs_image_dir = image_dir % coco_set_name
    image_name_template = 'abstract_v002_' + coco_set_name + '_%012d'
    #dataset = [None] * len(questions)
    dataset = []

    unk_ans_count = 0
    for n_q, q in enumerate(questions):
        if (n_q + 1) % 10000 == 0:
            print('processing %d / %d' % (n_q + 1, len(questions)))
        image_id = q['image_id']
        question_id = q['question_id']
        image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_name + '.png')
        question_str = q['question']
        question_tokens = tokenize(question_str)

        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      question_id=question_id,
                      question_str=question_str,
                      question_tokens=question_tokens)

        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)
            answer_dict = Counter(valid_answers)
            no_answer = True
            for i in answer_dict.values():
                if i >= 3:
                    no_answer = False
            if len(valid_answers) == 0 or no_answer:
                #valid_answers = ['<unk>']
                unk_ans_count += 1
                continue

            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers
            #print(iminfo['all_answers'])
            #print(iminfo['valid_answers'])
            #exit(1)

        #dataset[n_q] = iminfo
        dataset.append(iminfo)
    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
    return dataset


def main(args):
    image_dir = args.input_dir + '/%s/'
    annotation_file = args.input_dir + '/annotations/abstract_v002_%s_annotations.json'
    question_file = args.input_dir + '/questions/OpenEnded_abstract_v002_%s_questions.json'

    vocab_answer_file = args.output_dir + '/vocab_answers_train.txt'
    answer_dict = VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)
    #print(valid_answer_set)

    #train = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'train2015')
    valid = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'train2015')
    #test = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test2015')
    #test_dev = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test-dev2015')

    #np.save(args.output_dir + '/train.npy', np.array(train))
    print(valid[0])
    print(len(valid))
    np.save(args.output_dir + '/train.npy', np.array(valid))
    #np.save(args.output_dir + '/train_valid.npy', np.array(train + valid))
    #np.save(args.output_dir + '/test.npy', np.array(test))
    #np.save(args.output_dir + '/test-dev.npy', np.array(test_dev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./data',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='./datasets',
                        help='directory for outputs')

    args = parser.parse_args()

    main(args)
