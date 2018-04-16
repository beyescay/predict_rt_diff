from bs4 import BeautifulSoup
import time
import json
from contextlib import contextmanager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException


from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from fake_useragent import UserAgent






class text_to_be_not_present_in_element(object):
    """ An expectation for checking if the given text is present in the
    specified element.
    locator, text
    """
    def __init__(self, locator, text_):
        self.locator = locator
        self.text = text_

    def __call__(self, driver):
        try :
            element_text = EC._find_element(driver, self.locator).text
            return not(self.text in element_text)
        except StaleElementReferenceException:
            return False

def parse_movie_urls():
    '''Creates dictionary of all movies under Browse All of RT using div class = "mb-movie"
    and storing it as {title:url}'''

    #create dict
    movie_list = {}

    ua=UserAgent()
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = (ua.random)
    service_args=['--ssl-protocol=any','--ignore-ssl-errors=true']
    driver = webdriver.Chrome('/home/sarath/Documents/Shiva/rotten_tomatoes/chromedriver',desired_capabilities=dcap,service_args=service_args)

    #access website
    driver.get('https://www.rottentomatoes.com/browse/dvd-streaming-all/')

    soup = BeautifulSoup(driver.page_source, "lxml")
    showing_text_line = soup.find("div", {"id": "count-link"}).text

    total_count = int(showing_text_line.split()[3])

    print(total_count)

    elem = driver.find_element_by_xpath('//*[@id="show-more-btn"]/button')

    while(True):

        #check current count vs total count
        soup = BeautifulSoup(driver.page_source, "lxml")

        showing_text_line = soup.find("div", {"id": "count-link"}).text

        total_count = int(showing_text_line.split()[3])

        current_count = int(showing_text_line.split()[1])

        print(showing_text_line)

        #if it has finished clicking, break out of while loop
        if current_count >= total_count:
            break
        else:
            #continue clicking
            try:
                elem.click()
                wait = WebDriverWait(driver, 90)
                #wait until mb-movies is not stale
                # elem = wait.until((float(driver.find_element_by_class_name('mb-movies').get_attribute("style").split('opacity: ')[1][0]) == 0.5))
                # elem = wait.until((float(driver.find_element_by_class_name('mb-movies').get_attribute("style").split('opacity: ')[1][0]) == 1))
                wait.until(text_to_be_not_present_in_element((By.ID, "count-link"), showing_text_line))
            except ElementNotVisibleException:
                break
            except TimeoutException:
                break

    print('Scraping now')


def parse_html_page(html_page):

    movie_list = {}
    soup = BeautifulSoup(open(html_page), "lxml")

    #soup = BeautifulSoup(driver.page_source,"lxml")
    print soup
    movies = soup.findAll('div', {'class': "mb-movie"})
    print movies
    total_movies_count = len(movies)

    for idx, movie in enumerate(movies):
        print("Extracting movie {}/{}".format(idx+1, total_movies_count))
        print movie
        print("Prininting soup find")
        url = movie.find('a')['href']
        title = movie.find('h3',{'class' : "movieTitle"}).text
        movie_list[title] = url

    with open('movie_urls.txt','w') as file:
        json.dump(movie_list, file, indent=4)

    return 0



if __name__ == '__main__':
    #run scraper and print completion time
    print('Running')
    start = time.time()
    #movie_urls = parse_movie_urls()
    parse_html_page("9984_movies.html")
    end = time.time() - start
    print("Completed, time: " + str(end) + " secs")
