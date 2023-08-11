from bs4 import BeautifulSoup
from urllib.request import urlopen
from datetime import datetime
from collections import defaultdict
from googletrans import Translator as Translator1
from translate import Translator as Translator2
from deep_translator import GoogleTranslator as Translator
import nltk
import requests
import unicodedata
import psycopg2
import asyncio
import re
import os


nltk.download('averaged_perceptron_tagger')
nltk.download('words')

translator = Translator1()

API = "https://job.am/en/api/"
ENDPOINTS = ["jobs", "industry", "jobs/details/{Id}"]

ALL_INDUSTRIES = requests.get(API + ENDPOINTS[1]).json()

hostname = os.environ['DB_HOST']
database = os.environ['DB_NAME']
username = os.environ['DB_USER']
pwd = os.environ['DB_PASSWORD']
port_id = os.environ['DB_PORT']


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


def hy_to_en(hy_url):
    
    matches = re.match(r"(https://job.am)(.*)", hy_url)
    en_url = matches.group(1) + r'/en' + matches.group(2)
    return en_url


def add_empty_values(index, job_infos, keys:list[str]):
    for key in keys:
                                
        for _ in range(index - len(job_infos[key])):
                                    job_infos[key].append("")

async def parse_html(response):
    soup = BeautifulSoup(response, 'html.parser')
    return soup


def unicode_data(text):
    return unicodedata.normalize("NFKD", text)


async def empty_coroutine(value):
    return value


async def translate_to_english_async(text):
    # print(type(text))
    await asyncio.sleep(0.01)
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
    
    conn, cur = None, None

        
    try:
        conn = psycopg2.connect(
            
            host = hostname,
            dbname = database,
            user = username,
            password = pwd,
            port = port_id
        )
        
        cur = conn.cursor()
        
        create_script = """
                        CREATE TABLE IF NOT EXISTS job (                      
                            id uuid DEFAULT uuid_generate_v1() PRIMARY KEY,
                            title text,
                            job_url text,
                            img_url text,
                            company varchar(200),
                            sphere varchar(150),
                            location varchar(100),
                            address varchar (100),
                            employment varchar (150),
                            salary varchar(150),
                            responsibilities text,
                            requirements text,
                            schedule varchar(150),
                            experience varchar(100),
                            apply_link text,
                            job_email text,
                            notes text,
                            date_posted timestamp,
                            deadline timestamp,
                            active bool
                        );
        """
        
        cur.execute(create_script)
        
        get_urls = """
                        SELECT job_url
                        FROM job
                        WHERE active = True"""
        
        cur.execute(get_urls) 
        all_urls = [item[0] for item in cur.fetchall()]      
        
        conn.commit()
        job_infos = defaultdict(list)
            
        real_urls = [hy_to_en(item['Url']) for item in requests.get(API + ENDPOINTS[0]).json()]
        
        non_repetitive_elements = [x for x in all_urls if x not in real_urls]
        
        for job in requests.get(API + ENDPOINTS[0]).json():
            
            if hy_to_en(job["Url"]) in all_urls:
                continue
            job_infos["job_url"].append(hy_to_en(job["Url"]))
            
            print(job_infos["job_url"][-1])
            response = requests.get(job_infos["job_url"][-1]).content
            soup = await parse_html(response)
            other_infos = soup.find('div', {'class' : 'col-md-12 d-flex flex-wrap rowInfo'})
            
            for info_address in other_infos.find_all('div', {'class' : 'info-address'}):
                match info_address.find_all('span')[0].text.strip():
                    case 'Employment type:':
                        job_infos["employment"].append(info_address.find_all('span')[1].text.strip())
                    case 'Work schedule:':
                        job_infos["schedule"].append(info_address.find_all('span')[1].text.strip())
                    case 'Work experience:':
                        job_infos["experience"].append(info_address.find_all('span')[1].text.strip())
                    case 'Salary:':
                        job_infos["salary"].append(info_address.find_all('span')[1].text.strip())
            

            about = soup.find('div', {'class' : 'about-container job--descr'})
            
            
            all_tags = [{tag.name : tag.text} for tag in about.find_all(recursive=False)]
            all_keys = [list(item.keys())[0] for item in all_tags]
            
            indices = find_indices(all_keys, 'h3')
            tag_soup = about.find_all('h3', recursive=False)
            
            for i in range(len(indices)):
                if "պարտականություն" in tag_soup[i].text.lower() or "responsibilit" in tag_soup[i].text.lower(): 
                    if i!= len(indices)-1:
                        sub_tags = '\n'.join(list(item.values())[0] for item in all_tags[indices[i]+1 : indices[i+1]])
                    else:
                        sub_tags = '\n'.join(list(item.values())[0] for item in all_tags[indices[i]+1 : ])
                    
                    job_infos["responsibilities"].append(sub_tags.strip())   
                    
                    
                if "պահանջ" in tag_soup[i].text.lower() or "requirement" in tag_soup[i].text.lower(): 
                    if i!= len(indices)-1:
                        sub_tags = '\n'.join(list(item.values())[0] for item in all_tags[indices[i]+1 : indices[i+1]])
                    else:
                        sub_tags = '\n'.join(list(item.values())[0] for item in all_tags[indices[i]+1 : ])
                    
                    job_infos["requirements"].append(sub_tags.strip())
                    
                if "նշում" in tag_soup[i].text.lower() or "note" in tag_soup[i].text.lower(): 
                    if i!= len(indices)-1:
                        sub_tags = '\n'.join(list(item.values())[0] for item in all_tags[indices[i]+1 : indices[i+1]])
                    else:
                        sub_tags = '\n'.join(list(item.values())[0] for item in all_tags[indices[i]+1 : ])
                    
                    job_infos["notes"].append(sub_tags.strip())
                    
            add_empty_values(len(job_infos["job_url"]), job_infos, ["employment", "schedule", "experience", "salary", "responsibilities", "requirements", "notes"])
            
            
            job_infos["img_url"].append(job["Logo"])
            job_infos["title"].append(job["Title"])
            job_infos["company"].append(job["Company"])
            job_infos["location"].append(job["Location"])

            industry_names = [item["name"] for item in ALL_INDUSTRIES if item["id"] in job["IndustryIds"]]
                
            spheres = ', '.join(industry_names)
            job_infos["sphere"].append(spheres)
            
            
            job_info = requests.get(API + ENDPOINTS[2].format(Id=job['Id'])).json()
            # print(job_info)
            timestamp = re.search(r'[0-9]+', job_info["DateExpires"]).group(0)
            deadline = datetime.fromtimestamp(int(timestamp)/1000)

            job_infos["deadline"].append(str(deadline))
            
            
            timestamp = re.search(r'[0-9]+', job_info["DatePosted"]).group(0)
            date_posted = datetime.fromtimestamp(int(timestamp)/1000)
            
            job_infos["date_posted"].append(str(date_posted))
            job_infos["address"].append(job_info["Address"])
            job_infos["job_email"].append(job_info["EmailMask"])
            job_infos["apply_link"].append(job_info["ApplyLink"])
            job_infos["active"].append(1)
            
        # print(job_infos)

        if job_infos['job_url']:

            async def process_job_info_async(key, values):
                tasks = [translate_to_english_async(value) if value else empty_coroutine(value) for value in values]
                # print(tasks)
                translated_values = await asyncio.gather(*tasks)
                return key, translated_values

            async def process_job_infos_async(job_infos):
                coroutines = [process_job_info_async(key, values) for key, values in job_infos.items()]
                # print(coroutines)
                import tracemalloc
                tracemalloc.start()
                translated_job_infos = dict(await asyncio.gather(*coroutines))
                return translated_job_infos

            # Call the function to process job_infos asynchronously
            translated_job_infos = await process_job_infos_async(job_infos)

            
            job_infos = translated_job_infos
            

            value_text = ", "
            
            keys = str(tuple(job_infos.keys())) 
            
            values = value_text.join([str(tuple([unicode_data(value[index].replace("'", "’")) if value[index] else "NULL" for value in job_infos.values()])) for index in range(len(job_infos['job_url']))])
            
                
            keys = keys.replace("'", "")
            
            insert_script = f"""
                            INSERT INTO job {keys}
                            VALUES {values};
            """
            # print(insert_script)
            
            insert_script = insert_script.replace("'NULL'", 'NULL').replace("'NULL'", 'NULL')
            
            cur.execute(insert_script)
            
            keys = list(job_infos.keys())
            keys.remove('date_posted')
            keys.remove('deadline')
            keys.remove('active')
            
            for key in keys:
                update_script = f"""
                                UPDATE job
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
            
            # cur.execute("SELECT id, title FROM job")
            # rows = cur.fetchall()
            # for row in rows:
            #     record_id, old_value = row
            #     new_value = await translate_to_english_async(old_value)
            #     print(old_value, new_value)
            #     # Execute an UPDATE query to update the table
            #     cur.execute("UPDATE job SET title = %s WHERE id = %s", (new_value, record_id))
            
            conn.commit()
        
        # print(non_repetitive_elements)
        for url in non_repetitive_elements:
                change_script = f"""
                                UPDATE job
                                SET active = false
                                WHERE job_url = '{url}' ;"""
       
                cur.execute(change_script)
                
        change_script = f"""
                                UPDATE job
                                SET active = false
                                WHERE deadline <= '{datetime.now()}' ;"""
       
        cur.execute(change_script)
        
        conn.commit()     
        
    except Exception as error:
        print(error)

    finally:     
        
        if cur:
            cur.close()
            
        if conn:
            conn.close()


if __name__=="__main__":
    asyncio.run(parse())
