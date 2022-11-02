import os.path
import ssl
import urllib.parse
import urllib.request
import time
import datetime
from bs4 import BeautifulSoup
import sys

####################################################
# 여기 설정 변경이 필요합니다.

# 쿠키를 설정합니다.
cookie = "__cfduid=ddb52fe87ad8e727923cec214c775d3151557047972; _jp_visitor_token=cff7da02-eb18-4f70-9283-fe611eff1a8a; _jp_visit_token=d41ee5d3-c812-4175-9287-784238959bd7; _jp_traffic_source=%7B%22utm_campaign%22%3Anull%2C%22utm_medium%22%3Anull%2C%22utm_source%22%3Anull%7D; _jp_induction_status=true; cto_lwid=75c9f275-0727-415b-b599-ce4fa07102be; __auc=0a0a3d1a16a874ab98ed0699b69; _ga=GA1.3.1345685064.1557047983; _gid=GA1.3.25066355.1557047983; __gads=ID=2ae5d93c2efdac76:T=1557047976:S=ALNI_MYKgzykxgGUr6HcAAnIVCUKBNx-vw; _fbp=fb.2.1557047983491.1170496783; Jobplanet_remember_user_token=W1syMDU1MjU2XSwiJDJhJDEwJEh0Q1kyaFh3Z3hyTnduLm52VnFRNy4iXQ%3D%3D--4a76892cf79f6f69b49a9202d4485c194d267c76; premiumInfoModalOn=on; _jp_visit_short_token=1557051471805-ef601d6f-4db9-4b87-985c-51954427b3dd; __asc=f8ac1cc116a8781a52e9e08694e; wcs_bt=s_2f6e701a2a7e:1557051582; request_method=GET; _intween_x2o_net_session=Rlpaa0htNE5kazd6ZjFNSndNcmpNeUFSS21BdnN4b2daMHAyTUFvRnFvZStSZTYxa2JtWDFROFNzTUpVd2o4VmRWRmlWNm52a3lpeVZVcGRpN01tM1ZSNTY0RXErUGducHN5UmV6dUlZZlk0Vkl0M2dCMVVST1d4ZVBWTkNaenVERWNDcFpONktObnZiWXkvVFVvZjIxYTB3RHIzRXhBK2xJVGN5ZXZXalZnWXMrU3I0RDc1eGFjV25pTktDTnU1aFFNZWFpUEJVZnVlNGNUekhOUWNQaHVYWjVUZXBScXJsUnhNbFg1TzRGeWZJSmtRSmRuV3lyM0pLeER5K2JBOS0tV0VLKytGYWNjZmFCMHV5enQzK0dQdz09--83e43e65bac343d8fff929e625d1ddbc2f4e9096"

# 저장 폴더 경로를 지정합니다.
# 예 
# save_folder = "C:\\Users\\jin\\Documents\\data\\test\\"
save_folder = "D:\data crawling"

#출력할 산업 군을 리스트로 입력합니다

industry_list = [
    1000
    ]


# 지정한 숫자 이하의 회사 리뷰는 스킵합니다. 
review_min_count = 199

# 데이터 수집 날짜 구간을 지정합니다
# 예
# 19년 1월 1일 부터 19년 3월 1일까지의 데이터를 구하고 싶을때
# first_date = '20190101'
# last_date = '20190301'
first_date = ''
last_date = ''

# 여기 까지
#######################################################


first_time = 0
last_time = 0
if len(first_date) :
    first_time = time.mktime(datetime.datetime.strptime(first_date, "%Y%m%d").timetuple())
if len(last_date) :
    last_time = time.mktime(datetime.datetime.strptime(last_date, "%Y%m%d").timetuple())

if first_time > last_time :
    print("OPTION_ERROR. first_date is greater than end_date")
    sys.exit(1)
if len(industry_list) == 0 :
    print("OPTION_ERROR. industry_list is empty")
    sys.exit(1)

if not os.path.isdir(save_folder) :
    print("OPTION_ERROR. not found save folder")
    sys.exit(1)

if len(cookie) == 0 :
    print("OPTION_ERROR. cookie is empty")
    sys.exit(1)


test_loop_count=0
company_list_test_loop_count=0
list_url="https://www.jobplanet.co.kr/companies?industry_id="

def write_commend(date, company_name, company_id, comment_id, scores, recommend_str, up_str) :
    """
    구분자 : \t
    순서 : 코멘트 아이디, 승진 기회 및 가능성, 복지 및 급여, 업무와 삶의 균형, 사내문화, 경영진, 추천, 향후기대
    name : date, cname, cid, cmid, possibiilty, pay, balance, culture, management, recommend, expect
    value :
        date : 날
        cid : int
        possibility, pay, balance, culture, management :  int (1-5)
        recommend : None, 0(추천안함), 1(추천)
        expect: None, 0(하락), 1(비슷), 2(상승/성장)
    """
    
    file_name = "data{}_{}.csv".format(date.split("/")[0],date.split("/")[1])
    #print(file_name)
    if not os.path.isfile(save_folder+file_name) :
        print("Create File {}{}".format(save_folder,file_name))
        f = open(save_folder+file_name, 'w')
        f.write("date\tcname\tcid\tcmid\tpossibility\tpay\tbalance\tculture\tmanagement\trecommend\texpect\n")
    else :
        f = open(save_folder+file_name, 'a')

    vals = [date, company_name, company_id, comment_id]
    vals.extend(scores)

    recommend = None
    expect=None
    if recommend_str is not None :
        if recommend_str == "이 기업을 추천하지 않습니다.":
            recommend = 0
        elif recommend_str == "이 기업을 추천 합니다!":
            recommend = 1
        else :
            print("Check Recommend Invalid {}".format(recommend_str))
        
    if up_str is not None :
        if up_str == "하락" :
            expect = 0
        elif  up_str == "비슷" :
            expect = 1
        elif  up_str == "성장" :
            expect = 2
        else :
            print("Check Expect Invalid {}".format(up_str))
    vals.extend([recommend, expect])
    
    line = "\t".join(str(e) for e in vals) +"\n"
    f.write(line)
    
    f.close()


def getCompanyHttp(url):
    scontext = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        
    #req = urllib.request.Request(test_url, data=data.encode("utf-8"))
    req = urllib.request.Request(url)
    req.add_header("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36") 
    req.add_header("cookie", cookie)
    response = urllib.request.urlopen(req, context=scontext)
    result =  response.read().decode("utf-8")
    return result

def getCompanyCommends(url, page=1, list_page=1):

    cid = url.split("/")[4]
    cname = None
    
    while True :
        page_url = url
        print()
        if page > 1 :
            if page_url[-1:] == "?" :
                page_url+="page={}".format(page)
            else :
                page_url+="?page={}".format(page)
        print(page_url)
        
        html = getCompanyHttp(page_url)
    
        soup = BeautifulSoup(html, 'html.parser')

        if  cname is None :
            cname = soup.find("h2", {"class":"tit"}).text

        commends = soup.find_all("section",  {"class":"content_ty4" })

        commend_count = (len(commends))
        print("cname={} list_page ={} page ={} recommend_count = {}".format(cname, list_page, page, commend_count))
        if commend_count == 0 :
            break

        for commend in commends :
            cmid = commend["data-content_id"]
        
            values = []
            setdate = commend.find("span", {"class":"txt2"}).text
            settime = time.mktime(datetime.datetime.strptime(setdate, "%Y/%m/%d").timetuple())
            if first_time > 0 :
                if settime < first_time :
                    print("skip. date over. {}".format(setdate))
                    continue
            if last_time > 0 :
                if settime > last_time :
                    print("skip. yet date {}".format(setdate))
                    continue

            for pct in commend.find_all("div", {"class":"bl_score"}) :
                score=int(pct["style"].split(":")[1][:-2]) / 20
                values.append(score)
            is_up = commend.find("strong")
            recommend_str = commend.find("p",{"class":"recommend"})
            #dir(is_up)
            if is_up is not None :
                is_up = is_up.text
            if recommend_str is not None :
                recommend_str = recommend_str.text
            write_commend(setdate, cname, cid, cmid, values, recommend_str, is_up)
        
        if test_loop_count > 0 and test_loop_count <= page:
            break
        page+=1
        

for industry_id in industry_list :
    industry_list_url="{}{}".format(list_url, industry_id)
    print(industry_list_url)

    page=1
    while True :
        if page > 1 :
            html = getCompanyHttp("{}&page={}".format(industry_list_url, page))
        else :
            html = getCompanyHttp(industry_list_url)
    
        soup = BeautifulSoup(html, 'html.parser')
        companys = soup.find_all("section",  {"class":"content_ty3" })
        
        if len(companys) == 0 :
            break
        print("")
        print("Company List. page={} count={}".format(page, len(companys)))

        is_break=False
        for company in companys :
            company_name=company.find("dt", {"class":"us_titb_l3"}).find("a").text
            print(company_name)
            
            review_count_str = company.find("a", {"class":"us_stxt_1"}).text
            review_count = int(review_count_str.split(" ")[0])
            if review_count <= review_min_count:
                print("Pass Review Count 0. {} {}".format(company_name, review_count_str))
                is_break=True
                continue
            
            company_url=company.find("a", {"class":"us_stxt_1"})["href"]
            company_url="https://www.jobplanet.co.kr{}".format(company_url)
            #print(company_url)
            getCompanyCommends(company_url, list_page=page)
            
        if company_list_test_loop_count > 0 and company_list_test_loop_count <= page :
            break
        if is_break :
            break
        page+=1

        #test break
        #break

    
#url="https://www.jobplanet.co.kr/companies/88335/reviews/%EC%8F%98%EC%B9%B4"
#getCompanyCommends(url, page=2)
