from flask import Flask, jsonify, request, abort, render_template
from flask_caching import Cache
from tika import parser
import openai
from fuzzywuzzy import fuzz
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import sys
import os
import re
import asyncio
import numpy as np
import parsing.parse_quickstart as parse_course
import parsing.parse_job as parse_job
from googletrans import Translator as Translator1
from translate import Translator as Translator2
from deep_translator import GoogleTranslator as Translator
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stwords = stopwords.words('english')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

print("Course parsing...")
asyncio.run(parse_course.parse())
print("Job parsing...")
asyncio.run(parse_job.parse())

import spacy


sys.path.insert(0, "service_for_proflab")
from version import __version__, __description__


app = Flask(__name__)
cache = Cache(app)

app.config["CACHE_TYPE"] = "simple"  
# app.config["CACHE_DEFAULT_TIMEOUT"] = 300  


translator = Translator1()

MODEL1 = "gpt-3.5-turbo"
MODEL2 = "text-davinci-003"

openai.api_key = os.environ['API_KEY']

hostname = os.environ['DB_HOST']
database = os.environ['DB_NAME']
username = os.environ['DB_USER']
pwd = os.environ['DB_PASSWORD']
port_id = os.environ['DB_PORT']



@app.errorhandler(Exception)
def handle_error(error):
    # Get the traceback
    error_traceback = traceback.format_exc()
    if hasattr(error, 'code'):
        status_code = error.code
    else:
        status_code = 500
    return {"message" : error.description, "status_code" : status_code}, status_code
    
    
@app.after_request
def after_request(response):
  response.headers.set('Access-Control-Allow-Origin', '*')
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response 


@app.route("/", methods=["GET"])
def info():
    return __description__


def is_english(text):
    try:
        words = nltk.word_tokenize(text)
        english_words = set(nltk.corpus.words.words())
        
        # Check if all the words in the text are English words
        return all(word.lower() in english_words or not word.isalpha() for word in words)
    
    except: return False


def translate_to_english(text):
    # print(type(text))
    try:
        to_translate = str(text)
        return Translator(source='auto', target='en').translate(to_translate)
    except:
        try:
            result = translator.translate(str(text), dest='en')
            return result.text
        except : 
            try:
                translator= Translator2(to_lang="en")
                return translator.translate(str(text))
            except: return text


async def clean_questions(questions:dict):
    
    data = questions.copy()
    data['skills'] = ', '.join(questions['skills'])
    data['languages'] = ', '.join([item['language'] + ' - ' + item['proficiency'] for item in questions['languages']])
    data['licenses'] = ', '.join([item['title'] for item in questions['licenses']])
    data['educations'] = ', '.join([item['field'] + ' - ' + item['degree'] for item in questions['educations']])
    data['experiences'] = ', '.join([item['description'] + 'Position - ' + item['position'] for item in questions['experiences']])
    
    return data    

@app.route("/get_professions", methods=["POST"])
async def get_professions():
    #   ---Get file and parse content---
    
    # content = request.get_json()
    
    try:
        data = request.json['data']
        print(data)
    except: abort(500, "Invalid data of answers")

    # data = request.files['data']
    
    # schema = """Name (text)
    #             Birthday (date)
    #             Skills (text)
    #             Languages (text)
    #             Work Experience:
    #             -Company (text)
    #             -Position (text)
    #             -Start Date (date)
    #             -End Date (date)
    #             Education:
    #             -University (text)
    #             -Degree (text)
    #             -Start Date (date)
    #             -End Date (date)
    #             Appointment:
    #             -Company (text)
    #             -Position (text)
    #             -Start Date (date)
    #             -End Date (date)"""
    
    
    # parsed = parser.from_buffer(file.read())
    # text = parsed["content"]
    # text = re.sub('\n', ' ', text)
    
    # tokens = [token.strip() for token in text.split()]
    
    # cleaned_text = ''
    # for token in tokens:
    #     if token not in stwords:
    #         cleaned_text += token + ' '
    
    # content = re.sub(" {2,}", " ", cleaned_text)
    # content = content.strip()
    
    
    
       
    # main_prompt = f"""Parse CV content and translate to english- '{content}' based on '{schema}' """
    
    # try:
    #     completion = openai.Completion.create(
    #                     engine=MODEL2,
    #                     prompt=main_prompt,
    #                     temperature = 0.1 ** 100,
    #                     max_tokens = 1600
    #                 )
    # except openai.error.InvalidRequestError:
    #     return {'message' : """Tokens count passed""", 'status' : 'error'}
    
    # parsed_resume = (completion["choices"][0]["text"]).strip()
    
    
    # path = os.path.realpath(__file__)
    # dir = os.path.dirname(path)
    

    # answers_list = json.load(data)

    # answers = [{"question" : answer["question"], "answer" : answer["answers"][0]} if is_english(answer["answers"][0]) \
    #             else {"question" : answer["question"], "answer" : translate_to_english(answer["answers"][0])} for answer in answers_list]

    questions_one = data['userData']
    questions_two = data['questionAnswers']
    
    questions_two = [item if (isinstance(type(item['answer']), str)) else {'question' : item['question'], 'answer' : ', '.join(item['answer'])} for item in questions_two]

    questions_one = await clean_questions(questions_one)
    
    questions = list(questions_one.keys())
    answers = list(questions_one.values())
    
    questions.extend([item['question'] for item in questions_two])
    answers.extend([item['answer'] for item in questions_two])
    
    
    answers_data = [questions[i].strip() + ' : ' + answers[i] if await asyncio.get_event_loop().run_in_executor(None, is_english, answers[i]) \
                else questions[i].strip() + ' : ' +  await asyncio.get_event_loop().run_in_executor(None, translate_to_english, answers[i]) for i in range(len(questions))]
    
    
    
    answers_txt = ',\n'.join(answers_data[1:])
    # print(answers_txt)
    main_prompt = f"""
                    You are career coach, I am providing you information about career questions and answers. 
                    
                    You need determine the best match 4 professions for me. Presented professions should be and according to me, 
                    
                    trending, modern, perspective, independent of each other and, most importantly, did not coincide with my professions.
                    
                    The questions and answers: {answers_txt}.
                    
                    For each specialty, give me a short description and short rationale as to why it is appropriate. 
                    (Only use "you" application style when addressing me, do not apply by name.)
    """

    try:
        completion = openai.Completion.create(
                        engine=MODEL2,
                        prompt=main_prompt,
                        temperature = 0.1 ** 100,
                        max_tokens = 600
                    )
    except openai.error.InvalidRequestError:
            return {"message" : """Tokens count passed""", "status" : "error"}

    proffesions = (completion["choices"][0]["text"]).strip()
    
    proffesions = re.sub(r'\b(I|Me)\b', 'You', proffesions)
    proffesions = re.sub(r'\b(i|me)\b', 'you', proffesions)
    

    proffesions = re.sub(r'\bMy\b', 'Your', proffesions)
    proffesions = re.sub(r'\bmy\b', 'your', proffesions)
    
    profs_list = re.findall(r'[0-9]\. ([^\n]*)', proffesions.strip())
    
    profs = []
    for prof in profs_list:
        groups = re.match(r'(.*):(.*)', prof.strip()).groups()
        profs.append({'profession' : groups[0].strip(), 'about' : groups[1].strip()})
    
    return {"data": profs, "status" : 200}


async def get_courses(cur):
    
    get_courses_script = """
                        SELECT course_url, title, img_url, price, source, start_date, status
                        FROM course
                        WHERE active = TRUE
                        ORDER BY id ASC
                        limit 16
                        """
    cur.execute(get_courses_script)
    response = cur.fetchall()
    return [{k:v  for k, v in dict(item).items()}  for item in response]


async def get_jobs(cur):
    
    get_jobs_script = """
                        SELECT location, address, company, deadline, img_url, job_url, salary, title
                        FROM job
                        WHERE active = TRUE
                        ORDER BY date_posted DESC
                        limit 16
                        """
    cur.execute(get_jobs_script)
    response = cur.fetchall()
    return [{k:v  for k, v in dict(item).items()}  for item in response]

    
@app.route("/get_home_page", methods=["GET"])
async def get_home_page():
    
    conn, cur = None, None
        
        
    try:
        conn = psycopg2.connect(
            
            host = hostname,
            dbname = database,
            user = username,
            password = pwd,
            port = port_id
        )
        
        cur = conn.cursor(cursor_factory = RealDictCursor)
        
        course_response, job_response  = await asyncio.gather(get_courses(cur), get_jobs(cur))
        
    except psycopg2.Error as e: abort(str(e), e.pgcode)
      
    finally:     
        
        if cur:
            cur.close()
            
        if conn:
            conn.close()
    
    return {"courses" : course_response, "jobs" : job_response}



@app.route("/update_courses", methods=["GET"])
async def update_courses():
    
    try:
        await parse_job.parse()
        return {"status" : 200 , "message" : "Table 'course' is updated"}
    except Exception as e: abort(str(e), 500)
    

@app.route("/update_jobs", methods=["GET"])
async def update_jobs():
    
    try:
        await parse_course.parse()
        return {"status" : 200 , "message" : "Table 'job' is updated"}
    except Exception as e: abort(str(e), 500)
 

@app.route("/get_recommendation", methods=["POST"])
async def get_recommendation():
     
    try:
        profession = request.json["profession"]
    except: abort(500, "Invalid data of profession")
    
    try:
        data = request.json['data']
    except: abort(500, "Invalid data of answers")
    
    questions_one = data['userData']
    questions_two = data['questionAnswers']
    

    questions_one = await clean_questions(questions_one)
    
    questions = list(questions_one.keys())
    answers = list(questions_one.values())
     
    answers_data = [questions[i].strip() + ' : ' + answers[i] if await asyncio.get_event_loop().run_in_executor(None, is_english, answers[i]) \
                else questions[i].strip() + ' : ' +  await asyncio.get_event_loop().run_in_executor(None, translate_to_english, answers[i]) for i in range(len(questions))]
    
    
    main_prompt = f"Give required qualifications for this career path - {profession}. Rate the importance of each on a scale of 1-10 (Return in json form)"
    print(main_prompt)
    completion = openai.Completion.create(
                        engine=MODEL2,
                        prompt=main_prompt,
                        temperature = 0.1 ** 100,
                        max_tokens = 300
                    )
    

    skills = {k: v for k, v in sorted(eval((completion["choices"][0]["text"]).strip()).items(), key=lambda item: int(item[1]), reverse=True)}
    
    # file = request.files["file"]
    # parsed = parser.from_buffer(file.read())
    # text = parsed["content"]
    # text = re.sub('\n', ' ', text)
    
    # tokens = [token.strip() for token in text.split()]
    
    # cleaned_text = ''
    # for token in tokens:
    #     if token not in stwords:
    #         cleaned_text += token + ' '
    
    # content = re.sub(" {2,}", " ", cleaned_text)
    # content = content.strip()
    # print(', '.join(list(skills.keys())))

    
    pers_answers_txt = ',\n'.join(answers_data[1:8])
    prof_answers_txt = ',\n'.join(answers_data[8:])
    

    questions = [item['question'] for item in questions_two]
    answers = [item['answer'] for item in questions_two]
    
    answers_data = [questions[i].strip() + ' : ' + answers[i] if await asyncio.get_event_loop().run_in_executor(None, is_english, answers[i]) \
                else questions[i].strip() + ' : ' +  await asyncio.get_event_loop().run_in_executor(None, translate_to_english, answers[i]) for i in range(len(questions))]
    
    
        
    psych_answers_txt = ',\n'.join(answers_data)
    
    print(answers_data)
    
    # print(answers_data)

    
    
    
    main_prompt = f"""
                    You are candidate coach. I answered 3 types of questions. Here are the questions and my answers:
                          1. personal questions - {pers_answers_txt},
                          2. professional questions - {prof_answers_txt},
                          3. psychological questions - {psych_answers_txt}.
                    Based on my answers, please analyze and determine how well it fits the requirements for each of these skills: {', '.join(list(skills.keys()))} (In the following format - "Each skill in the following list ({', '.join(list(skills.keys()))}) : some text").
                    Rate it very strictly on a scale of 0 to 10. Break down each component of the rating and briefly explain why you assigned that particular value.
                    Also, give me a suggestion (in the following format - "Suggestion: Some Text") about what skills i need to improve or develop for a better fit and therefore a higher score. (Do not give an overall score.)
                    (Only use "you" application style when addressing me, do not apply by name.)
                    """
    
    try:

        completion = openai.Completion.create(
                        engine=MODEL2,
                        prompt=main_prompt,
                        temperature = 0.1 ** 100,
                        max_tokens = 600
                    )
        
    except openai.error.InvalidRequestError:
            abort("Tokens count passed",  403)

    text = (completion["choices"][0]["text"]).strip()
    # print(text)
    
    text = re.sub(r'\b(I am|Me am)\b', 'You are', text)
    text = re.sub(r'\b(i am|me am)\b', 'you are', text)
    
    text = re.sub(r'\b(I|Me)\b', 'You', text)
    text = re.sub(r'\b(i|me)\b', 'you', text)
    

    text = re.sub(r'\bMy\b', 'Your', text)
    text = re.sub(r'\bmy\b', 'your', text)
    
    # print(text)
    
    score_patern = r'^(.*?):(\s*\d+)'
    
    matches = re.findall(score_patern, text, re.MULTILINE)

    scores_dict = {match[0]: int(match[1].strip()) for match in matches}
    
    total = sum(skills.values())
    
    score = 0
    
    for skill in skills:
        score += (100/total) * (scores_dict[skill]/10) * skills[skill]
    
    suggestion_pattern = r':\s*([^:]+)'
    
    match = re.findall(suggestion_pattern, text, re.MULTILINE)[-1]
    
    suggestion = match.strip()
    
    
    evaluation_pattern = r'^.*?:\s*[0-9]*'
    
    evaluation = re.findall(evaluation_pattern, text, re.MULTILINE)[:-1]

    data = {}
    for line in evaluation:
        key, value = line.split(':')
        data[key] = int(value.strip())
    
    with open(r"C:\Users\user\Desktop\evaluation.json", "w") as file:    
        json.dump(data, file)
        
    evaluation_pattern = r'^.*?:.*'
    
    evaluation = re.findall(evaluation_pattern, text, re.MULTILINE)[:-1]

    score_data = []
    skill_data = []
    weights = {}
    
    for line in evaluation:
        key, value = line.split(':')
        groups = re.match(r'([0-9]+)/*[0-9]* -(.*)', value.strip())
        score_data.append({'title' : key, 'value' :  int(groups.group(1).strip()), 'evaluation' : groups.group(2).strip()})
    
        weights[key] = int(groups.group(1).strip())

    for skill, value in skills.items():
        skill_data.append({'title' : skill, 'value' : value})
    
    # return ({"evaluation" : score_data, "total_score" : round(score,1), "suggestion" : suggestion, "skills" : skill_data})
    
    conn, cur = None, None
        
    try:
        conn = psycopg2.connect(
            
            host = hostname,
            dbname = database,
            user = username,
            password = pwd,
            port = port_id
        )
        
        cur = conn.cursor(cursor_factory = RealDictCursor)
        
        courses_jobs = await asyncio.gather(get_rec_courses(cur, profession, skills, weights), get_rec_jobs(cur, profession, skills, weights))

        # print(courses_jobs)
       
    except psycopg2.Error as e: abort(str(e), e.pgcode)

    finally:     
        
        if cur:
            cur.close()
            
        if conn:
            conn.close()
    
    return ({"evaluation" : score_data, "total_score" : round(score,1), "suggestion" : suggestion, "skills" : skill_data, "recommendation" : courses_jobs, "status" : 200})
     
     
async def get_rec_courses(cur, profession, skills, weights):
    
    get_script = """
                    SELECT id, title, sphere, description
                    FROM course
                    WHERE active = TRUE;
                    """
    
    
    cur.execute(get_script)
    response = cur.fetchall()
    response = [{k:v  for k, v in dict(item).items()}  for item in response]
    
    # low_skills  = {k:v for k, v in skills.items() if v < 8}
    all_skills = {k:v for k, v in skills.items()}
    all_cleaned_skills = []
    
    all_tokens = [[token.strip() for token in (profession + ' - ' + skill).split()] for skill in all_skills]

    for tokens in all_tokens:
        cleaned_skill = ''
        for token in tokens:
            if token not in stwords:
                cleaned_skill += token + ' '
        all_cleaned_skills.append(cleaned_skill.strip())        

    nlp = spacy.load("en_core_web_md")

    courses = [item for item in response]
    
    
    matches = {}
    # courses.append({"title" : "Tableau for beginners", "sphere" : "IT", "description" : ""})

    profession_title = nlp(profession)
    for course in courses:
        course_title = nlp(course['sphere'] + " : " + course['title'].replace('AI', 'Artificial Intelligence'))
        # description = nlp(course['description'].replace('AI', 'Artificial Intelligence'))
        
        tokens = [token.strip() for token in (course['sphere'] + ' : ' + course['title'].replace('AI', 'Artificial Intelligence')).split()]
        cleaned_course = ''
        for token in tokens:
            if token not in stwords:
                cleaned_course += token + ' '
                
        similarity = []
        title_similarity = course_title.similarity(profession_title)
        for i, cleaned_skill in enumerate(all_cleaned_skills):
            # skill_name = nlp(list(all_skills.keys())[i])

            
            similarity.append(fuzz.partial_token_set_ratio(cleaned_course, cleaned_skill.replace("communication", "communication English, Russian").replace(\
                            "Communication", "Communication English, Russian"))/100 * (10 - skills[list(all_skills.keys())[i]]) * title_similarity**0.5 * weights[list(all_skills.keys())[i]])
            
            # print(similarity[-1], cleaned_course, cleaned_skill.replace("communication", "communication : English, Russian").replace("Communication", "Communication : English Russian"))

        matches[course['id']] = max(similarity)
        
    required_courses = [item[0] for item in sorted(matches.items(), key=lambda x: x[1], reverse=True)]
    # print(required_courses)
    
    courses = required_courses[:5]   
    
    course_ids = str(tuple(required_courses[:5]))
    # print(course_ids)
    
    get_script = f"""
                    SELECT course_url, title, img_url, price, source, start_date, status
                    FROM course
                    WHERE id in {course_ids};
                    """
    
    
    cur.execute(get_script)
    response = cur.fetchall()
    
    return {"courses" : [{k:v  for k, v in dict(item).items()}  for item in response]}


async def get_rec_jobs(cur, profession, skills, weights):
    get_script = """
                    SELECT id, title, sphere, requirements
                    FROM job
                    WHERE active = TRUE;
                    """
    
    
    cur.execute(get_script)
    response = cur.fetchall()
    response = [{k:v  for k, v in dict(item).items()}  for item in response]
    
    # low_skills  = {k:v for k, v in skills.items() if v < 8}
    all_skills = {k:v for k, v in skills.items()}
    all_cleaned_skills = []
    
    all_tokens = [[token.strip() for token in (profession + ' - ' + skill).split()] for skill in all_skills]

    for tokens in all_tokens:
        cleaned_skill = ''
        for token in tokens:
            if token not in stwords:
                cleaned_skill += token + ' '
        all_cleaned_skills.append(cleaned_skill.strip())        
                
    nlp = spacy.load("en_core_web_md")

    jobs = [item for item in response]

    matches = {}
    # courses.append({"title" : "Tableau for beginners", "sphere" : "IT", "description" : ""})

    profession_title = nlp(profession)
    
    for job in jobs:
        
        job_title = nlp(job['sphere'] + " : " + job['title'].replace('AI', 'Artificial Intelligence'))
        # if job['requirements']:
        #     requirements = nlp(job['requirements'].replace('AI', 'Artificial Intelligence'))
        # else: requirements = ''
        similarity = []
        
        tokens = [token.strip() for token in (job['sphere'] + ' : ' + job['title'].replace('AI', 'Artificial Intelligence')).split()]
        cleaned_job = ''
        for token in tokens:
            if token not in stwords:
                cleaned_job += token + ' '
        
        fuzz_ratios = np.vectorize(fuzz.partial_token_set_ratio)([skill.replace("communication", "communication English, Russian").replace(\
                            "Communication", "Communication English, Russian") for skill in all_skills.keys()], cleaned_job)
        job_similarity = job_title.similarity(profession_title)
        
        for i, cleaned_skill in enumerate(all_cleaned_skills):
            # skill_name = nlp(cleaned_skill)


            # if requirements:
            similarity.append(fuzz_ratios[i]/100 * (skills[list(all_skills.keys())[i]]) * job_similarity**0.5 *\
                        weights[list(all_skills.keys())[i]])
            # else: similarity.append(fuzz_ratios[i]/100 * (skills[list(all_skills.keys())[i]]) * job_similarity**0.5 *\
            #                 weights[list(all_skills.keys())[i]])
            
            if isinstance(similarity[-1], complex):
                similarity.pop()
            # print(similarity[-1], cleaned_job, requirements)

        matches[job['id']] = max(similarity)
        
    required_jobs = [item[0] for item in sorted(matches.items(), key=lambda x: x[1], reverse=True)]
    # print(required_jobs)

    jobs = required_jobs[:5]   
    
    job_ids = str(tuple(required_jobs[:5]))
    # print(job_ids)
    
    get_script = f"""
                    SELECT location, address, company, deadline, img_url, job_url, salary, title
                    FROM job
                    WHERE id in {job_ids};
                    """
    
    
    cur.execute(get_script)
    response = cur.fetchall()

    return {"jobs" : [{k:v  for k, v in dict(item).items()}  for item in response]}
            