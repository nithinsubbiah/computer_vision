from external.vqa.vqa import VQA

annotation_json_file_path = './DATA/mscoco_train2014_annotations.json'
question_json_file_path = './DATA/OpenEnded_mscoco_train2014_questions.json'

vqa = VQA(annotation_json_file_path, question_json_file_path)

print(len(vqa.getQuesIds()))

q_id = 409380

question = vqa.loadQA(q_id)

