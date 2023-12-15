from flask import Flask, jsonify, request, abort
import sys
sys.path.append('service_for_proflab')
from config.config import Config
from tika import parser
import openai
from fuzzywuzzy import fuzz
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import re
import asyncio
import spacy
from flask_apscheduler import APScheduler
# import parsing.parse_quickstart as parse_course
import parsing.parse_job as parse_job
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stwords = stopwords.words('english')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')


sys.path.insert(0, "service_for_proflab")
from version import __version__, __description__


app = Flask(__name__)


MODEL1 = "gpt-3.5-turbo"
MODEL2 = "text-davinci-003"

API_DOCS = Config.API_DOCS

openai.api_key = Config.OPENAI_KEY
UDEMY_KEY = Config.UDEMY_KEY
COURSES_URL = Config.COURSES_URL

hostname = Config.DATABASE_HOST
database = Config.DATABASE_NAME
username = Config.DATABASE_USER
pwd = Config.DATABASE_PASSWORD
port_id = Config.DATABASE_PORT


hostname = "localhost"
database = "flask_db"
username = "postgres"
pwd = "Tik.555"
port_id = 5432


import logging

logging.basicConfig(level=logging.INFO)  # You can adjust the logging level as needed
logger = logging.getLogger(__name__)

scheduler = APScheduler()

def job():
    asyncio.run(update_courses_jobs())


@app.errorhandler(Exception)
def handle_error(error):
    # Get the traceback
    error_traceback = traceback.format_exc()
    print(error_traceback)
    if hasattr(error, 'code'):
        status_code = error.code
    else:
        status_code = 500
    return {"message" : str(error), "status" : status_code}, status_code
    
    
@app.after_request
def after_request(response):
  response.headers.set('Access-Control-Allow-Origin', '*')
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response 


@app.route("/", methods=["GET"])
async def info():
    return __description__


# def is_english(text):
#     try:
#         words = nltk.word_tokenize(text)
#         english_words = set(nltk.corpus.words.words())
        
#         # Check if all the words in the text are English words
#         return all(word.lower() in english_words or not word.isalpha() for word in words)
    
#     except: return False


# def translate_to_english(text):
#     # print(type(text))
#     try:
#         to_translate = str(text)
#         return Translator(source='auto', target='en').translate(to_translate)
#     except:
#         try:
#             result = translator.translate(str(text), dest='en')
#             return result.text
#         except : 
#             try:
#                 translator= Translator2(to_lang="en")
#                 return translator.translate(str(text))
#             except: return text


async def replace_none_with_empty(data):
    return [{key: val if val is not None else "" for key, val in item.items()} for item in data]


async def clean_questions(questions:dict):
    
    data = questions.copy()
    data['skills'] = ', '.join(questions['skills'])
    updated_dict = await replace_none_with_empty(questions['languages'])
    data['languages'] = ', '.join([item['language'] + ' - ' + item['proficiency'] for item in updated_dict])
    updated_dict = await replace_none_with_empty(questions['licenses'])
    data['licenses'] = ', '.join([item['title'] for item in updated_dict])
    updated_dict = await replace_none_with_empty(questions['trainings'])
    data['trainings'] = ', '.join([item['description'] + ', Position - ' + item['title'] for item in updated_dict])
    updated_dict = await replace_none_with_empty(questions['educations'])
    data['educations'] = ', '.join(['Field - ' + item['field'] + ', Degree - ' + item['degree'] for item in updated_dict])
    updated_dict = await replace_none_with_empty(questions['experiences'])
    data['experiences'] = ', '.join([item['description'] + ', Position - ' + item['position'] for item in updated_dict])
    
    return data    


async def check_token_valid(token):
    response = requests.post(API_DOCS + '/api/auth/service-token/check', json={'token' : token})
    if response.status_code != 200:
        abort(response.status_code, response.content.decode('utf-8'))
    return response.json()['valid']
       


@app.route("/get_professions", methods=["POST"])
async def get_professions():
    
    
    try:
        authorization_header = request.headers.get('Authorization')
        print(authorization_header)
    except: abort(500, "Invalid authorization header")
        
    if authorization_header:
        _, token = authorization_header.split()
        
        if not await check_token_valid(token):
            abort(403, "Invalid authorization token")
    
    else: abort(401, "Authorization header not found")
    
    
    try:
        data = request.json['data']
        # print(data)
    except: abort(400, "Invalid data of answers")
    

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
    
    questions_two = [item if (isinstance(type(item['answers']), str)) else {'question' : item['question'], 'answers' : ', '.join(item['answers'])} for item in questions_two]
    
    exclude_professions = [item['field'] for item  in questions_one['educations']]
    exclude_professions.extend([item['position'] for item  in questions_one['experiences']])
    
    # print(exclude_professions)
    questions_one = await clean_questions(questions_one)
    
    questions = list(questions_one.keys())
    answers = list(questions_one.values())
    
    questions.extend([item['question'] for item in questions_two])
    answers.extend([item['answers'] for item in questions_two])

    
    answers_data = [questions[i].strip() + ' : ' + answers[i].strip() if answers[i] else questions[i].strip() + ' : ' for i in range(len(questions))]
    
    pers_answers_txt = ',\n'.join(answers_data[1:8]).strip()
    prof_answers_txt = ',\n'.join(answers_data[8:14]).strip()
    
    # print(prof_answers_txt)
    # questions_two = [item if (isinstance(type(item['answers']), str)) else {'question' : item['question'], 'answers' : ', '.join(item['answers'])} for item in questions_two]
    
    questions = [item['question'] for item in questions_two]
    answers = [item['answers'] for item in questions_two]
    
    answers_data = [questions[i].strip() + ' : ' + answers[i] if answers[i] else questions[i].strip() + ' : ' for i in range(len(questions))]
    
    
        
    psych_answers_txt = ',\n'.join(answers_data).strip()
    
    
    # answers_txt = ',\n'.join(answers_data[1:])
    
    # print(prof_answers_txt)
    
    main_prompt = f"""
                    Analyze the user’s personal information, education, experience, background, answers to the test questions and based on this 
                    information suggest four new different professions that best correspond to this personality. 
                    The suggested professions should not repeat the professions from education and experience. 
                    Provide description of each suggested profession and explanation for choosing each profession.

                    Here are the questions and my answers (Translate to english if needed):
                          1. personal questions - {pers_answers_txt},
                          2. professional questions - {prof_answers_txt},
                          3. psychological questions - {psych_answers_txt}.

                    (Only use "you" application style when addressing me, do not apply by name.)
    """

    try:
        completion = openai.Completion.create(
                        engine=MODEL2,
                        prompt=main_prompt,
                        temperature = 0,
                        max_tokens = 600
                    )

    except Exception as e:
        return {"message" : str(e), "status" : 520}
    
    # print(completion)
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


@app.route('/get_courses', methods=['GET', 'POST'])
async def get_courses():

    if request.method == 'POST':
        try:
            data = request.json
            search = data.get('search', '')
            size = data.get('size', 16)
                
        except: abort(400, "Invalid data of courses")
    
    if request.method == 'GET':
        search = ''
        size = 16
    
    url = "https://www.udemy.com/api-2.0/courses/"
    headers = {
    'Accept': 'application/json, text/plain, */*',
    'Authorization': 'Basic SXpuNExVN1Y0dE83VlphY3R0WG5WMkgxOXNIOWd3Z2hDY25xa2xtZjpKZTZkTGNUeWV5MGs0U2JBT1FtQmFGRXBqZVQ2UHJNQVVPb1pyOHZxdlVEWWM3aFdFV0RxY3pPdHJXRnF4b21uSWlyd2tSektHWEVuc3FPZjd1cUFjVEpkOVFRc1JTaEJKRDAxTGdVcFBNT0JvdVRxVmV4UWNUWVlvTzNuMWJwbA==',
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    params = {
        'page_size': size,  # Number of results to retrieve per page
        'page': 1,  # Page number to retrieve
        'ordering': 'relevance'  # Order by relevance
    }
    
    if search:
        params['search'] = search
        
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return {"courses" : [{'title' : item.get('title', ''), 'course_url' : 'https://www.udemy.com' + item.get('url', ''), 'price' : item.get('price', ''), 'source' : 'Udemy', 
                    'img_url' : item.get('image_480x270', ''), 'status' : 'Online'}
                    for item in response.json().get('results', [])]}
        else:
            # print(response.__dict__, type(UDEMY_KEY))
            return([])
    except Exception as e: abort(response.status_code, str(e))
    
    # conn, cur = None, None
        
        
    # try:
        
    #     conn = psycopg2.connect(
            
    #         host = hostname,
    #         dbname = database,
    #         user = username,
    #         password = pwd,
    #         port = port_id
    #     )
        
    #     cur = conn.cursor(cursor_factory = RealDictCursor)
        
    #     get_courses_script = """
    #                         SELECT course_url, title, img_url, price, source, start_date, status
    #                         FROM course
    #                         WHERE active = TRUE
    #                         ORDER BY parse_date DESC
    #                         limit 16
    #                         """
    #     cur.execute(get_courses_script)
    #     response = cur.fetchall()
    #     return [{k:v  for k, v in dict(item).items()}  for item in response]

    # except psycopg2.OperationalError as e:  abort(500, "Error connecting to the database: " + str(e))
    # except Exception as e: abort(500, str(e))
    
    # finally:     
        
    #     if cur:
    #         cur.close()
            
    #     if conn:
    #         conn.close()
    
   
            
async def get_jobs():
    
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
        
        get_jobs_script = """
                            SELECT location, address, company, deadline, img_url, job_url, salary, title
                            FROM job
                            WHERE active = TRUE
                            ORDER BY date_posted DESC
                            limit 16
                            """
        cur.execute(get_jobs_script)
        response = cur.fetchall()
    
    except psycopg2.OperationalError as e:  abort(500, "Error connecting to the database: " + str(e))
    except Exception as e: abort(500, str(e))
    
    finally:     
        
        if cur:
            cur.close()
            
        if conn:
            conn.close()
            
    return {"jobs" : [{k:v  for k, v in dict(item).items()}  for item in response]}

    
@app.route("/get_home_page", methods=["GET"])
async def get_home_page():
        
        tasks = [get_jobs(), call_courses_url()]
        
        course_response, job_response  = await asyncio.gather(*tasks)
    
        return {**course_response, **job_response}



# @app.route("/update", methods=["GET"])
async def update_courses_jobs():
    
    try:
        logger.info("Course and Job tables creating or/and updating...")
        # await parse_course.parse()
        await parse_job.parse()
    except psycopg2.OperationalError as e: abort(500, "Error connecting to the database: " + str(e))
    except Exception as e: abort(500, str(e))
    
    return {"status" : 200, "message" : "'Course' and 'Job' tables are updated"}
 

@app.route("/get_recommendation", methods=["POST"])
async def get_recommendation():
     
    try:
        authorization_header = request.headers.get('Authorization')
    except: abort(500, "Invalid authorization header")
        
    if authorization_header:
        _, token = authorization_header.split()
        
        if not await check_token_valid(token):
            abort(403, "Invalid authorization token")
    
    else: abort(401, "Authorization header not found")


    try:
        profession = request.json["profession"]
    except: abort(400, "Invalid data of profession")
    
    try:
        data = request.json['data']
    except: abort(400, "Invalid data of answers")
    
    questions_one = data['userData']
    questions_two = data['questionAnswers']
    

    questions_one = await clean_questions(questions_one)
    
    questions = list(questions_one.keys())
    answers = list(questions_one.values())
     
    logger.info(questions_one)
     
    answers_data = [questions[i].strip() + ' : ' + answers[i].strip() if answers[i] else questions[i].strip() + ' : ' for i in range(len(questions))]
    
    
    main_prompt = f"Give required qualifications for this career path - {profession}. Rate the importance of each on a scale of 1-10 (Return in json form)"

    completion = openai.Completion.create(
                        engine=MODEL2,
                        prompt=main_prompt,
                        temperature = 0,
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

    
    pers_answers_txt = ',\n'.join(answers_data[1:8]).strip()
    prof_answers_txt = ',\n'.join(answers_data[8:14]).strip()
    
    questions_two = [item if (isinstance(type(item['answers']), str)) else {'question' : item['question'], 'answers' : ', '.join(item['answers'])} for item in questions_two]
    
    questions = [item['question'] for item in questions_two]
    answers = [item['answers'] for item in questions_two]
    
    answers_data = [questions[i].strip() + ' : ' + answers[i] if answers[i] else questions[i].strip() + ' : ' for i in range(len(questions))]
    
    
        
    psych_answers_txt = ',\n'.join(answers_data).strip()
    
    
    main_prompt = f"""
                    You are candidate coach. I answered 3 types of questions. Here are the questions and my answers (Translate to english if needed):
                          1. personal questions - {pers_answers_txt},
                          2. professional questions - {prof_answers_txt},
                          3. psychological questions - {psych_answers_txt}.
                    Based on my answers, please analyze and determine how well it fits the requirements for each of these skills: {', '.join(list(skills.keys()))} (On the following format - "Each skill in the following list ({', '.join(list(skills.keys()))}) : some text").
                    Rate it very strictly on a scale of 0 to 10. Break down each component of the rating and briefly explain why you assigned that particular value.
                    Also, give me a suggestion (in the following format - "Suggestion: Some Text") about what skills i need to improve or develop for a better fit and therefore a higher score. (Do not give an overall score.)
                    (Only use "you" application style when addressing me, do not apply by name.)
                    """
    
    try:

        completion = openai.Completion.create(
                        engine=MODEL2,
                        prompt=main_prompt,
                        temperature = 0,
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

    scores_dict =  {match[0]: int(match[1].strip()) if match[0] in skills.keys() else 5 for match in matches}

    
    total = sum(skills.values())
    
    score = 0
    
    for skill in skills:
        score += (100/total) * (scores_dict.get(skill, 5)/10) * skills[skill]
    
    suggestion_pattern = r':\s*([^:]+)'
    
    match = re.findall(suggestion_pattern, text, re.MULTILINE)[-1]
    
    suggestion = match.strip()
    

        
    evaluation_pattern = r'^.*?:.*'
    
    evaluation = re.findall(evaluation_pattern, text, re.MULTILINE)[:-1]

    score_data = []
    skill_data = []
    weights = {}
    
    # print(evaluation)
    
    for line in evaluation:
        key, value = line.split(':')
        if key in skills.keys():
            groups = re.match(r'([0-9]+)/*[0-9]*\s*-(.*)', value.strip())
            if groups:
                score_data.append({'title' : key, 'value' :  int(groups.group(1).strip()), 'evaluation' : groups.group(2).strip()})
                    
                weights[key] = int(groups.group(1).strip())
                
            else: 
                score_data.append({'title' : key, 'value' :  5, 'evaluation' : ""})
                weights[key] = 5
                
    for skill, value in skills.items():
        skill_data.append({'title' : skill.strip(), 'value' : value})
    
    # return ({"evaluation" : score_data, "total_score" : round(score,1), "suggestion" : suggestion, "skills" : skill_data})
    
    tasks = get_rec_courses(profession, skills, weights), get_rec_jobs(profession, skills, weights)
    
    get_courses = await asyncio.gather(*tasks)
    
    return {"evaluation" : score_data, "total_score" : round(score,1), "suggestion" : suggestion, "skills" : skill_data, "recommendation" : {**get_courses[0], **get_courses[1]}, "status" : 200}


# import yappi
# from functools import wraps
# import asyncio

# def async_profile(fn):
#     @wraps(fn)
#     async def profiled(*args, **kwargs):
#         yappi.set_clock_type("cpu")
#         yappi.start()
#         try:
#             result = await fn(*args, **kwargs)
#         finally:
#             yappi.stop()
#             yappi.get_func_stats().print_all()
#         return result
#     return profiled


async def call_courses_url(search=None, size=None):
    try:    
        if not search and not size:
            response = requests.get(COURSES_URL)
            response.raise_for_status()
            
            return response.json()
        else:
            response = requests.post(COURSES_URL, data={'search' : search, 'size': size})
            response.raise_for_status()
            
            return response.json()
    
    except requests.HTTPError as e:
        return {"courses" : []}
        abort(e.response.status_code, e.response.text)
    except Exception as e:
        return {"courses" : []}
        abort(408, str(e))

@app.route("/get_courses_jobs", methods=["POST"])
async def get_courses_jobs():   
    
    try:
        authorization_header = request.headers.get('Authorization')
    except: abort(500, "Invalid authorization header")
        
    if authorization_header:
        _, token = authorization_header.split()
        
        if not await check_token_valid(token):
            abort(403, "Invalid authorization token")
    
    else: abort(401, "Authorization header not found")
    
    try:
        profession = request.json["profession"]
    except: abort(400, "Invalid data of profession")
    
    try:
        skills_data = request.json['skills']
        
        skills = {item['title'] : item['value'] for item in skills_data}
        
    except: abort(400, "Invalid data of skills")
    
    try:
        evaluation = request.json['evaluation']
        
        weights = {item['title'] : item['value'] for item in evaluation}
                  
    except: abort(400, "Invalid data of evaluation")
    
    print("Calling rec_courses...")
    
    tasks = [get_rec_courses(profession, skills, weights), get_rec_jobs(profession, skills, weights)]
    get_courses = await asyncio.gather(*tasks)       
  
    
    return {"recommendation" : {**get_courses[0], **get_courses[1]}, "evaluation" : evaluation,  "skills" : skills_data, "status" : 200}
   
    
# course_cache = {}

async def get_rec_courses(profession, skills, weights):
    
    return await call_courses_url(search=profession, size=5)

#     try:
        
#         conn, cur = None, None
        
#         # Check if the result is cached

#         conn = psycopg2.connect(
#                 host=hostname,
#                 dbname=database,
#                 user=username,
#                 password=pwd,
#                 port=port_id
#             )

#         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

#         get_script = """
#                         SELECT id, title, sphere, description
#                         FROM course
#                         WHERE active = TRUE;
#                         """

#         cur.execute(get_script)

#         response = cur.fetchall()

#         courses = [item for item in response]

#         cache_key = (profession, tuple(weights.items()), tuple(skills.items()), tuple(item['id'] for item in response))
        
        
#         if cache_key in course_cache:
#             response = course_cache[cache_key]
            
#         else:

#             nlp = spacy.load("en_core_web_md")
#             # Perform NLP operations on the profession once and reuse
#             profession_title = nlp(profession)

#             # Filter stopwords and tokenize skills once and reuse
#             all_tokens = [
#                 [token.strip() for token in (profession + ' - ' + skill).split()]
#                 for skill in skills
#             ]

#             all_cleaned_skills = [
#                 ' '.join(token for token in tokens if token not in stwords)
#                 for tokens in all_tokens
#             ]


#             matches = {}
            

#             for course in courses:
#                 # Perform NLP operations on the course title once and reuse
#                 course_t = course['sphere'] + " : " + course['title'].replace('AI', 'Artificial Intelligence')
#                 course_title = nlp(course_t)

#                 cleaned_course = ' '.join(token for token in course_t.split() if token not in stwords)

#                 title_similarity = course_title.similarity(profession_title)

#                 # Use list comprehension to calculate similarity
#                 similarity = [
#                     (fuzz.partial_token_set_ratio(cleaned_course, re.sub(r'[Cc]ommunication', "Communication English, Russian", cleaned_skill))
#                     / 100 * (10 - skills[list(skills.keys())[i]]) * title_similarity ** 0.5 * weights.get(list(skills.keys())[i], 5))
#                     for i, cleaned_skill in enumerate(all_cleaned_skills)
#                 ]

#                 matches[course['id']] = max(similarity)

#             required_courses = [item[0] for item in sorted(matches.items(), key=lambda x: x[1], reverse=True)]
            
            
#             course_ids = str(tuple(required_courses[:5]))
#             # print(course_ids)
            
#             get_script = f"""
#                             SELECT course_url, title, img_url, price, source, start_date, status
#                             FROM course
#                             WHERE id in {course_ids};
#                             """
            
            
#             cur.execute(get_script)
#             response = cur.fetchall()
#             # Cache the result for future use
            
#             dict_size_bytes = sys.getsizeof(course_cache)
#             dict_size_mb = dict_size_bytes / (1024 * 1024)
            
#             while dict_size_mb >=256:
#                 first_key = next(iter(course_cache))
#                 del course_cache[first_key]
#                 dict_size_bytes = sys.getsizeof(course_cache)
#                 dict_size_mb = dict_size_bytes / (1024 * 1024)
                
#             course_cache[cache_key] = response

#     except psycopg2.OperationalError as e:
#         return jsonify({"error": "Error connecting to the database: " + str(e)}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()
            
#     # print(response, required_courses)
    
#     return {"courses" : [{k:v  for k, v in dict(item).items()}  for item in response]}


job_cache = {}


async def get_rec_jobs(profession, skills, weights):
    
    
    try:
        
        conn, cur = None, None
        
        conn = psycopg2.connect(
            
            host = hostname,
            dbname = database,
            user = username,
            password = pwd,
            port = port_id
        )
        
        cur = conn.cursor(cursor_factory = RealDictCursor)
            
        cache_key = (profession, tuple(weights.items()), tuple(skills.items()))
        
        if cache_key in job_cache:
            response = job_cache[cache_key]
            
        else:
            get_script = """
                            SELECT id, title, sphere
                            FROM job
                            WHERE active = TRUE;
                            """
            
            
            cur.execute(get_script)
            response = cur.fetchall()
            response = [{k:v  for k, v in dict(item).items()}  for item in response]
            
            # low_skills  = {k:v for k, v in skills.items() if v < 8}
            
            all_tokens = [
                [token.strip() for token in (profession + ' - ' + skill).split()]
                for skill in skills
            ]

            all_cleaned_skills = [
                ' '.join(token for token in tokens if token not in stwords)
                for tokens in all_tokens
            ]
                        
            # nlp = spacy.load("en_core_web_md")

            jobs = [item for item in response]

            matches = {}
            # courses.append({"title" : "Tableau for beginners", "sphere" : "IT", "description" : ""})

            
            for job in jobs:
                job_t = job['sphere'] + " : " + job['title'].replace('AI', 'Artificial Intelligence')

                
                cleaned_job = ' '.join(token for token in job_t.split() if token not in stwords)

                # job_similarity = job_title.similarity(profession_title)
                

                similarity= [(fuzz.partial_token_set_ratio(cleaned_job, re.sub(r'[Cc]ommunication', "Communication English, Russian", cleaned_skill))/
                              100 * fuzz.partial_token_set_ratio(job['title'].replace('AI', 'Artificial Intelligence'), profession)/100 * weights.get(list(skills.keys())[i], 5))
                              for i, cleaned_skill in enumerate(all_cleaned_skills)]
                  
                if isinstance(similarity[-1], complex):
                    similarity.pop()
                    
                # if job['id'] == '72cd41ce-4891-11ee-9918-7c10c98c62c3':
                #     print([[(cleaned_job, re.sub(r'[Cc]ommunication', "Communication English, Russian", cleaned_skill)), (fuzz.partial_ratio(cleaned_job, re.sub(r'[Cc]ommunication', "Communication English, Russian", cleaned_skill))/
                #               100 * fuzz.token_sort_ratio(job['title'].replace('AI', 'Artificial Intelligence'), profession)/100 ), weights.get(list(skills.keys())[i], 5)]
                #               for i, cleaned_skill in enumerate(all_cleaned_skills)])
                #     similarity= [(fuzz.partial_ratio(cleaned_job, re.sub(r'[Cc]ommunication', "Communication English, Russian", cleaned_skill))/
                #               100 * fuzz.token_sort_ratio(job['title'].replace('AI', 'Artificial Intelligence'), profession)/100 * weights.get(list(skills.keys())[i], 5))
                #               for i, cleaned_skill in enumerate(all_cleaned_skills)]
                #     print(similarity)
                    
                matches[job['id']] = max(similarity)
                
            # print([item[0] for item in sorted(matches.items(), key=lambda x: x[1], reverse=True)])
            required_jobs = [item[0] for item in sorted(matches.items(), key=lambda x: x[1], reverse=True)]
            # print(required_jobs) 
                
            job_ids = str(tuple(required_jobs[:5]))
            # print(str(tuple(required_jobs[:10])))
            
            get_script = f"""
                            SELECT location, address, company, deadline, img_url, job_url, salary, title
                            FROM job
                            WHERE id in {job_ids};
                            """
            
            
            cur.execute(get_script)
            response = cur.fetchall()
            
            dict_size_bytes = sys.getsizeof(job_cache)
            dict_size_mb = dict_size_bytes / (1024 * 1024)
            
            while dict_size_mb >=256:
                first_key = next(iter(job_cache))
                del job_cache[first_key]
                dict_size_bytes = sys.getsizeof(job_cache)
                dict_size_mb = dict_size_bytes / (1024 * 1024)

            job_cache[cache_key] = response

    except psycopg2.OperationalError as e:  abort(500, "Error connecting to the database: " + str(e))
    except Exception as e: abort(500, str(e))
    
    finally:     
        
        if cur:
            cur.close()
            
        if conn:
            conn.close()
            
    return {"jobs" : [{k:v  for k, v in dict(item).items()}  for item in response]}
