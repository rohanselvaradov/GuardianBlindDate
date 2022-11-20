import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

# Fewer than 50 pages of blind dates but filters duplicates
INDEX_URLS = ['https://www.theguardian.com/lifeandstyle/series/blind-date?page=' + str(i + 1) for i in range(50)]
PAGES_DIR = "pages"
POSITIVE_PHRASES = ["yes", "absolutely", "!", "definitely"]
NEGATIVE_PHRASES = ["no", "probably not", "not sure", "as friends"]
NEUTRAL_PRONOUNS = ["they", "them", "theirs"]
FEMALE_PRONOUNS = ["she", "her", "hers"]
MALE_PRONOUNS = ["he", "him", "his"]
PRONOUNS = NEUTRAL_PRONOUNS + FEMALE_PRONOUNS + MALE_PRONOUNS
MINIMUM_Q_ANSWERS = 100
NUMBER_MAPPINGS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
}


def collect_urls(index_urls):
    """Collects all the urls from the index pages"""
    page_urls = []
    for url in index_urls:
        index_page = requests.get(url)  # Get the index page
        if index_page.status_code != 200 or index_page.url != url:  # Check page loaded correctly and did not redirect
            continue
        soup = BeautifulSoup(index_page.text, 'html.parser')  # Parse the page
        page_urls += [l.a.get('href') for l in soup.find_all('div', class_='fc-item__container')]  # Add links to list
        print("Collected {} links successfully".format(
            len(page_urls)))  # Print total number of links collected parsed so far
    print()
    with open('page_urls.txt', 'w') as f:  # Save the list of links to a file
        for url in page_urls:
            f.write(url + "\n")
    return page_urls


def collect_pages(page_urls, data=None):
    if data is None:
        data = {}
    if not os.path.exists(PAGES_DIR):
        os.mkdir(PAGES_DIR)
    failures = []
    i = 0
    saved = 0
    loaded = 0
    for page_url in page_urls:
        page_id = re.sub('/', '--', re.split('.com/', page_url)[-1])
        temp_path = os.path.join(os.getcwd(), PAGES_DIR, '{}.html'.format(page_id))
        if os.path.exists(temp_path):
            loaded += 1
        else:
            page_data = requests.get(page_url)
            if page_data.status_code != 200 or page_data.url != page_url:
                failures.append(page_url)
                continue
            open(temp_path, 'w', encoding='utf-8').write(page_data.text)  # Write the page to file
            saved += 1
        data[page_id] = {}
        data[page_id]['url'] = page_url
        if i % 100 == 0 and i != 0:
            print("Parsed {} pages".format(i))
        i += 1
    print(
        "\nParsed {} pages total:\n\t{} already on file;\n\t{} downloaded and saved to file.".format(i, loaded, saved))
    print("Unsuccessfully attempted to fetch {} pages.\n".format(len(failures)))
    if len(failures) != 0 and input("Show failed pages? (y/n): ").lower() == 'y':
        for f in failures:
            print(f)
    if i + len(failures) != len(page_urls):
        print("Warning: did not attempt to fetch {} pages".format(len(page_urls) - i - len(failures)))
    return data, failures


def parse_pages(data):
    i = 0
    failures = []
    for page in os.listdir(PAGES_DIR):
        page_id = re.sub('.html', '', page)
        page_path = os.path.join(os.getcwd(), PAGES_DIR, page)
        with open(page_path, 'r', encoding='utf-8') as f:  # maybe add encoding='utf-8'
            soup = BeautifulSoup(f.read(), 'html.parser')
        if 'celebrity' in soup.title.text.lower():  # Skip celebrity blind dates
            continue
        try:
            find_1 = soup.find(id=re.compile('-on-'))
            if find_1 is None:
                find_2 = soup.find('h3', class_='dcr-18sg7f2', string=re.compile(' on '))
                person_a, person_b = find_2.text.lower().split(' on ')
            else:
                person_a, person_b = find_1.attrs['id'].split('-on-')
        except AttributeError:  # Skip pages with no 'PERSON A on PERSON B' heading
            find_3 = page_id.split('--')[-1].split('-')
            if len(find_3) == 4:
                person_a, person_b = find_3[2:]
            else:
                failures.append(page_id)
                continue
        data[page_id]['date'] = pd.to_datetime(soup.find('meta', property='article:published_time')['content']).date()
        data[page_id]['person_a'] = person_a
        data[page_id]['person_b'] = person_b
        questions = set()
        for p in soup.find_all('p', class_='dcr-18sg7f2'):
            strongs = p.find_all('strong')
            if len(strongs) == 1:  # TODO - sort out for when there are multiple strongs e.g. https://www.theguardian.com/lifeandstyle/2022/oct/29/blind-date-maddy-jessie
                question = strongs[0].text  # Question
                clean_question = re.sub('[\W]+', '',
                                        question.strip().replace(" ", "_")).strip().lower()  # No punctuation or spaces
                generalised_question = re.sub(
                    '|'.join(['_' + x for x in PRONOUNS + [person_a.lower(), person_b.lower()]]),
                    '_X', clean_question)  # Generalise question
                prefix = "A_" if generalised_question not in questions else "B_"
                questions.add(generalised_question)
                answer = p.text[len(question):]
                data[page_id][prefix + generalised_question] = answer
        if i % 100 == 0 and i != 0:
            print("Extracted data from {} pages".format(i))
        i += 1

    if len(failures) != 0:
        print("\nFailed to parse {} pages".format(len(failures)))
        if input("Show failed pages? (y/n): ").lower() == 'y':
            for f in failures:
                print(f)
    print("\nExtracted data from {} pages total".format(i))
    return pd.DataFrame.from_dict(data, orient='index')


def process_data(data):
    new_data = data.copy(deep=True)

    new_data['A_marks_out_of_10_int'] = new_data['A_marks_out_of_10'].str.lower().replace(NUMBER_MAPPINGS, regex=True)
    new_data['B_marks_out_of_10_int'] = new_data['B_marks_out_of_10'].str.lower().replace(NUMBER_MAPPINGS, regex=True)
    new_data['A_marks_out_of_10_float'] = new_data['A_marks_out_of_10'].str.extract('(\d+\.\d+|\d+)').astype(
        float)  # Extract the number from the string
    new_data['A_marks_out_of_10_float'].fillna(new_data['A_marks_out_of_10_int'],
                                               inplace=True)  # Fill in missing values with the integer version
    new_data['A_marks_out_of_10_check'] = new_data['A_marks_out_of_10'].str.contains(
        '\d[\w\s/]+\d')  # Check for multiple numbers
    new_data['B_marks_out_of_10_float'] = new_data['B_marks_out_of_10'].str.extract('(\d+\.\d+|\d+)').astype(float)
    new_data['B_marks_out_of_10_float'].fillna(new_data['B_marks_out_of_10_int'], inplace=True)
    new_data['B_marks_out_of_10_check'] = new_data['B_marks_out_of_10'].str.contains('\d[\w\s/]+\d')

    new_data['A_would_you_meet_again_yes'] = new_data['A_would_you_meet_again'].str.contains(
        r'\b(?:{})\b'.format('|'.join(POSITIVE_PHRASES)), case=False)
    new_data['A_would_you_meet_again_no'] = new_data['A_would_you_meet_again'].str.contains(
        r'\b(?:{})\b'.format('|'.join(NEGATIVE_PHRASES)), case=False)
    new_data['A_would_you_meet_again_check'] = new_data['A_would_you_meet_again_yes'] == new_data[
        'A_would_you_meet_again_no']
    new_data['B_would_you_meet_again_yes'] = new_data['B_would_you_meet_again'].str.contains(
        r'\b(?:{})\b'.format('|'.join(POSITIVE_PHRASES)), case=False)
    new_data['B_would_you_meet_again_no'] = new_data['B_would_you_meet_again'].str.contains(
        r'\b(?:{})\b'.format('|'.join(NEGATIVE_PHRASES)), case=False)
    new_data['B_would_you_meet_again_check'] = new_data['B_would_you_meet_again_yes'] == new_data[
        'B_would_you_meet_again_no']

    new_data.dropna(thresh=MINIMUM_Q_ANSWERS, axis=1,
                    inplace=True)  # Drop columns with less than MINIMUM_Q_ANSWERS answers

    return new_data
    #  TODO - infer gender


def main():
    try:
        page_urls = [l.strip() for l in open("page_urls.txt", "r").readlines()]
    except FileNotFoundError:
        page_urls = collect_urls(INDEX_URLS)
    interim_data, failures = collect_pages(page_urls)
    while len(failures) != 0 and input("Retry failed pages? (y/n): ").lower() == 'y':
        interim_data, failures = collect_pages(failures, interim_data)
    parsed_data = parse_pages(interim_data)
    parsed_data.to_csv('parsed_data.csv', encoding='utf-8-sig', index=False)
    processed_data = process_data(parsed_data)
    processed_data.to_csv('processed_data.csv', encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    main()
