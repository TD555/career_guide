from bs4 import BeautifulSoup
from urllib.request import urlopen
from collections import defaultdict
import psycopg2
import asyncio
import unicodedata
import nltk
import os
from googletrans import Translator as Translator1
from translate import Translator as Translator2
from deep_translator import GoogleTranslator as Translator
import calendar
from datetime import datetime, date
import re
from config.config import Config


nltk.download('averaged_perceptron_tagger')
nltk.download('words')

translator = Translator1()
month_names = list(calendar.month_name)[1:]

hostname = Config.DATABASE_HOST
database = Config.DATABASE_NAME
username = Config.DATABASE_USER
pwd = Config.DATABASE_PASSWORD
port_id = Config.DATABASE_PORT


translator = Translator1()

#Parse required fields of for Quicstart courses

async def parse_html(response):
    soup = BeautifulSoup(response, 'html.parser')
    return soup


def unicode_data(text):
    return unicodedata.normalize("NFKD", text)


def match_pattern(pattern, text, method=re.match):
    match = method(pattern, text)
    if match:
        print(f"match {match.group(1)} -  {match.group(2)}")
        return  match.group(2)


def add_empty_values(index, course_infos, keys:list[str]):
    for key in keys:                             
        for _ in range(index - len(course_infos[key])):
                                    course_infos[key].append("")
       

async def empty_coroutine(value):
    return value


async def translate_to_english_async(text):
    print(type(text))
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

async def parse():
    
    course_urls = []
    real_urls = []
    course_infos = defaultdict(list)
    conn, cur = None, None

        
        
    
    conn = psycopg2.connect(
        
        host = hostname,
        dbname = database,
        user = username,
        password = pwd,
        port = port_id
    )
    
    cur = conn.cursor()
    
    create_script = """
                    CREATE TABLE IF NOT EXISTS course (                      
                        id uuid DEFAULT uuid_generate_v1() PRIMARY KEY,
                        title text,
                        course_url text,
                        img_url text,
                        sphere varchar(100),
                        description text,
                        training_requirements text,
                        schedule varchar(150),
                        duration varchar(150),
                        start_date DATE,
                        end_date DATE,
                        price text,
                        trainer text,
                        parse_date timestamp,
                        status varchar(50),
                        source varchar(100),
                        active bool
                    );
    """
    
    cur.execute(create_script)   
    conn.commit()
    
    get_urls = """
                    SELECT course_url
                    FROM course
                    WHERE active = TRUE"""
    
    cur.execute(get_urls) 
    
    all_urls = [item[0] for item in cur.fetchall()]      

    
 
    with urlopen('https://quickstart.am/en/open-trainings/#') as response:
        soup = await parse_html(response)
        exists =[]
        
        for anchor in list(soup.find_all('a', {"class": "eltdf-cli-link eltdf-block-drag-link"}))[::-1]:
            course_url = (anchor.get('href', '/'))
            if course_url not in all_urls:
                exists.append(True)
                course_infos["course_url"].append(course_url)
                course_infos["parse_date"].append(datetime.utcnow())
                await asyncio.sleep(0.001)
            else:
                real_urls.append(course_url) 
                exists.append(False)
                
        
        images = [img.img['src'] if img else "" for img in list(soup.find_all('div', {"class" : "eltdf-cli-image"}))[::-1]]
        for i in range(len(exists)):
            if exists[i]:
                course_infos["img_url"] .append(images[i])
            
            
        for i, anchor in enumerate(list(soup.find_all('div', {"class": "eltdf-cli-category-holder"}))[::-1]):
            if exists[i]:
                texts = []
                for sphere in anchor.find_all('a', {"class" : "eltdf-cli-category"}):
                    texts.append(sphere.text)
                # print(', '.join(texts), '\n')
                course_infos["sphere"].append(', '.join(texts))
        
    course_infos["source"] = len(course_infos["course_url"]) * ['Quick Start']
    course_infos["end_date"] = len(course_infos["course_url"]) * ["NULL"]

    
    non_repetitive_elements = [x for x in all_urls if x not in real_urls]
    
    
    course_urls = course_infos["course_url"]
    
    for course_url in course_urls:
        print(course_url)
        # if course_url != "https://quickstart.am/en/course/cv-writing-interview/":
        #     continue
        course_infos["active"].append(1)
        with urlopen(course_url) as response:
            print(course_urls.index(course_url), course_url)

            await asyncio.sleep(2)

            try: 
                soup = await parse_html(response)
            except Exception as error: 
                print(error)
                continue
                
            # await asyncio.sleep(1)
            
            title = soup.find('h2', {"class" : "eltdf-course-single-title"})
            if title: 
                course_infos["title"].append(title.text.strip())
            else: course_infos["title"].append("")
            
            # await asyncio.sleep(1)
            
            try:
                content = soup.find_all('div', {'class' : 'wpb_text_column wpb_content_element'})[-1].find('div')
            except: 
                content = None
                print(f"Empty content - {course_url}")
            
            all_tags = content.find_all()
            
            
            # for tag in all_tags:
                # print(content.find(tag.name))
            if content:
                # course_infos["content"].append(content.text.strip().replace("\'", "\’"))
                all_tags = [tag for tag in content.find_all(recursive=False)]
                
                add_empty_values(course_urls.index(course_url), course_infos, keys=list(course_infos.keys()))
                
                match_status = match_pattern(r'(Ձևաչափ՝)[\n]*(.+)', content.text.strip(), re.search)
                if match_status:
                    course_infos["status"].append(match_status.strip())      
                else:  course_infos["status"].append("offline")     
                
                course_infos["description"].append("") 
                
                added = False
                duration_added = False
                requirements_added = False
                schedule_added = False
                price_added = False
                trainer_added = False
                
                for index, tag in enumerate(all_tags):
                    
                    
                    match_duration = match_pattern(r'([Dd]*uration)[\n]*(.+)', tag.text.strip())
                    match_requirements = match_pattern(r'([Tt]*raining requirements)[\n]*(.+)', tag.text.strip())
                    match_schedule = match_pattern(r'([Ss]*chedule)[\n]*(.+)', tag.text.strip())
                    match_price = match_pattern(r'([Yy]*our [Ii]*nvestment)[\n]*(.+)', tag.text.strip())
                    match_trainer = match_pattern(r'([Tt]*rainer[s]*)[\n]*(.*)', tag.text.strip())
                    
                    if match_duration  and not duration_added:
                        course_infos["duration"].append(match_duration)
                        duration_added = True
                        added = True
                    
                    if match_requirements  and not requirements_added:
                        course_infos["training_requirements"].append(match_requirements)
                        requirements_added = True
                        added = True
                    
                    if match_schedule  and not schedule_added:
                        course_infos["schedule"].append(match_schedule)
                        schedule_added = True
                        added = True
                    
                    if match_price  and not price_added:
                        course_infos["price"].append(match_price)
                        price_added = True
                        added = True
                        
                    if match_trainer and not trainer_added:
                        course_infos["trainer"].append(match_trainer)
                        trainer_added = True
                        added = True
                        
                        
                    if tag.text.strip().lower() == "training requirements":
                        course_infos["training_requirements"].append(all_tags[index+1].text.strip())
                        added = True
                        requirements_added = True
                        print("training_requirements - ", all_tags[index+1].text.strip())
                            
                    elif tag.text.strip().lower() == 'schedule':
                        # if "duration" not in all_tags[index+1].text.strip().lower():
                            if "տևողություն" in all_tags[index+1].text.strip().lower():
                                added = True
                                duration_added = True
                                course_infos["duration"].append(all_tags[index+1].text.strip())
                                print("duration - ", all_tags[index+1].text.strip())
                            elif "օֆլայն" in all_tags[index+1].text.strip().lower() or "offline" in all_tags[index+1].text.strip().lower() \
                                or "օնլայն" in all_tags[index+1].text.strip().lower() or "online" in all_tags[index+1].text.strip().lower():
                                pass    
                            else: 
                                course_infos["schedule"].append(all_tags[index+1].text.strip())
                                schedule_added = True
                                added = True
                                print("schedule - ", all_tags[index+1].text.strip())
                                
                    elif tag.text.strip().lower() == 'duration':
                        course_infos["duration"].append(all_tags[index+1].text.strip())
                        added = True
                        duration_added = True
                        print("duration - ", all_tags[index+1].text.strip())
                                
                    elif tag.text.strip().lower() in ['your investment']:
                        course_infos["price"].append(all_tags[index+1].text.strip())
                        added = True
                        price_added = True
                        print("price - ", all_tags[index+1].text.strip())
                                
                    elif tag.text.strip().lower() in ["trainer", "trainers", "speaker", "խոսնակներ", "սեմինարը կվարի", "փորձագետ"]:
                        if all_tags[index+1].text.strip():
                            course_infos["trainer"].append(all_tags[index+1].text.strip())
                        else: course_infos["trainer"].append(all_tags[index+2].text.strip())
                        trainer_added = True
                        added = True
                        print("trainer - ", course_infos["trainer"][-1])
                    
                    elif not added and not match_pattern(r'(Ձևաչափ՝)[\n]*(.+)', tag.text.strip(), re.search):
                        course_infos["description"][-1] += "\n" + tag.text.strip()
                        added = False

            
        # print("description - ", course_infos["description"][-1])
        
    if course_infos['course_url']:

        add_empty_values(len(course_infos["course_url"]), course_infos, keys=list(course_infos.keys()))

        
        async def process_course_info_async(key, values):
            tasks = [translate_to_english_async(value) if value and key not in ("course_url", "img_url", "start_date", "end_date", "parse_date", "source", "active")
                                                       else empty_coroutine(value) for value in values]
            # print(tasks)
            translated_values = await asyncio.gather(*tasks)
            return key, translated_values

        async def process_course_infos_async(course_infos):
            coroutines = [process_course_info_async(key, values) for key, values in course_infos.items()]
            # print(coroutines)
            import tracemalloc
            tracemalloc.start()
            translated_course_infos = dict(await asyncio.gather(*coroutines))
            return translated_course_infos

        # Call the function to process course_infos asynchronously
        translated_course_infos = await process_course_infos_async(course_infos)
        
        course_infos = translated_course_infos
        
        
        print(course_infos["course_url"])
        print(course_infos["trainer"])
        
        course_infos["start_date"] = []
        
        for schedule in course_infos.get("schedule", []):
            match = re.search(r"([A-Za-z]+) (\d{1,2})", schedule)
            if match:
                if match.group(1) in month_names:
                    month = match.group(1)
                    day = match.group(2)
                    year = date.today().year
                    
                    month_num = datetime.strptime(month, "%B").month
                    
                    today = date.today()
                    
                    # Create a datetime object with the provided components
                    if month_num > today.month - 6:
                        date_obj = datetime(year, month_num, int(day))
                        
                    else: date_obj = datetime(year+1, month_num, int(day))

                    # Format the date as "DD/MM/YYYY"
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                    
                    course_infos["start_date"].append(formatted_date)
                    
                else: course_infos["start_date"].append("NULL")
            else: course_infos["start_date"].append("NULL")
        
        print("date : ", course_infos["start_date"][-1])
                                            
        value_text = ", "

        keys = str(tuple(course_infos.keys())) 

        values = value_text.join([str(tuple([unicode_data(str(value[index]).replace("'", "’")) if value[index] else "NULL" for value in course_infos.values()])) for index in range(len(course_infos['course_url']))])
        
        keys = keys.replace("'", "")
        
        insert_script = f"""
                        INSERT INTO course {keys}
                        VALUES {values};
        """
        # print(insert_script)
        
        insert_script = insert_script.replace("'NULL'", 'NULL')
        
        cur.execute(insert_script)
        conn.commit()
        
        keys = list(course_infos.keys())
        keys.remove('end_date')
        keys.remove('start_date')
        keys.remove('parse_date')
        keys.remove('active')
        
        for key in keys:
            update_script = f"""
                            UPDATE course
                            SET {key} = 
                            CASE
                                WHEN {key} LIKE '- %' THEN REPLACE({key}, '- ', '')
                                WHEN {key} LIKE '– %' THEN REPLACE({key}, '– ', '')
                                WHEN {key} LIKE ': %' THEN REPLACE({key}, ': ', '')
                                WHEN {key} LIKE '%’%' THEN REPLACE({key}, '’', '''')
                                ELSE {key}
                            END
                            WHERE {key} LIKE '- %' OR {key} LIKE ': %' OR {key} LIKE '– %' OR {key} LIKE '%’%';
                            """           
            cur.execute(update_script)
            conn.commit()
            
    # print(non_repetitive_elements)
    
    for url in non_repetitive_elements:
            change_script = f"""
                            UPDATE course
                            SET active = false
                            WHERE course_url = '{url}';"""

            cur.execute(change_script)
            conn.commit()
                    
    change_script = f"""
                            UPDATE course
                            SET active = false
                            WHERE START_DATE < '{date.today()}';"""
    
    cur.execute(change_script)
    conn.commit()     
    
    if cur:
        cur.close()
        
    if conn:
        conn.close()

if __name__ == "__main__":
    asyncio.run(parse())
    