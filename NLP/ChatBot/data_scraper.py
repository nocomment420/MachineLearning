from bs4 import BeautifulSoup, NavigableString, Tag
import requests


def load_doc(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    return soup


# --- basic english
def get_basic_english_urls(main_doc):
    urls = []
    section = main_doc.find_all("div", attrs={"class": "tcb-flex-row tcb--cols--3"})
    cols = section[0].children
    for col in cols:
        links = col.find_all("a")
        for link in links:
            urls.append(link.attrs["href"])

    return urls


def extract_basic_english_converation(url):
    doc = load_doc(url)
    print("loading converation: {}".format(doc.title))
    section = doc.find_all("div", attrs={"class": "tve_shortcode_rendered"})
    if len(section) == 0:
        section = doc.find_all("div", attrs={"class": "awr"})
        if len(section) == 0:
            print("error occured")
            return [], []
    questions = []
    answers = []
    is_question = True
    current = ""
    for child in section[0].children:
        # check for part of sentence
        if type(child) == NavigableString:
            if child != "\n" and child != "" and child != " ":
                current += str(child)

        # check for end of sentence
        elif type(child) == Tag:
            if "class" in child.attrs and "sc_player_container1" in child.attrs["class"]:
                if is_question:
                    questions.append(current)
                else:
                    answers.append(current)
                is_question = not is_question
                current = ""

    assert len(questions) == len(answers)
    print("Found {} question and answer pairs\n".format(len(questions)))
    return questions, answers


def get_basic_english_converation_data(url):
    main_doc = load_doc(url)
    urls = get_basic_english_urls(main_doc)
    print("loaded main docs, start to parse {} conversation links \n".format(len(urls)))

    questions = []
    answers = []
    for url in urls:
        q, a = extract_basic_english_converation(url)
        questions += q
        answers += a

    assert len(questions) == len(answers)
    print("finished loading converations, found {} questions and answers".format(len(questions)))

    return questions, answers


# ----- esl fast
def get_eslfast_urls(base_urk, index_url, domain, is_head):
    # load doc
    doc = load_doc(index_url)
    section = doc.find_all("section", attrs={"class": "beginners inner-page"})
    assert len(section) == 1
    if not is_head:
        print("Loaded document {}".format(doc.title))
    else:
        print("loaded head doccument, starting to extract children urls...")

    containers = section[0].find_all("div", attrs={"class": "list-page-container"})
    urls = []
    for container in containers:
        links = container.find_all("a")
        for link in links:
            if is_head:
                urls.append(base_urk + "/" + link.attrs["href"])
            else:
                urls.append(link.attrs["href"].replace("../..", domain))
    if not is_head:
        print("found {} urls\n".format(len(urls)))
        return urls
    else:
        print("found {} children urls, starting to extract secondary urls...\n".format(len(urls)))
        all_urls = []
        for url in urls:
            all_urls += get_eslfast_urls(base_urk, url, domain, False)
        print("Finished, found {} total urls\n".format(len(all_urls)))
        return all_urls


def get_elsfast_data(base_url, index_url, domain):
    urls = get_eslfast_urls(base_url, index_url, domain, True)
    # urls = ["https://www.eslfast.com/easydialogs/ec/dating06.htm"]

    questions = []
    answers = []

    print("Starting to extract questions and answers from urls...")
    for url in urls:
        is_sentence = False
        senteces = []

        doc = load_doc(url)
        section = doc.find_all("p", attrs={"class": "timed"})
        assert len(section) == 1

        for child in section[0].children:
            # check if next child is a sentence
            if type(child) == Tag and child.name == "b":
                is_sentence = True

            # if is sentence add to question or answer
            elif is_sentence and type(child) == NavigableString:
                senteces.append(str(child))
                is_sentence = False
        q = []
        a = []
        for i in range(len(senteces) - 1):
            q.append(senteces[i])
            a.append(senteces[i + 1])
        print("finished extracting questions and answers from {}, found {} questions and answers".format(doc.title,
                                                                                                         len(q)))
        questions += q
        answers += a

    print("Found {} questions and answers\n".format(len(questions)))
    return questions, answers


# ---- ello
def get_ello_urls(url, base):
    doc = load_doc(url)
    sections = doc.find_all("div", attrs={"class": "floatleftpro"})
    urls = []
    for section in sections:
        title = section.find_all("div", attrs={"class": "pro-image"})
        assert len(title) == 1
        link = title[0].find_all("a")
        if len(link) == 1:
            urls.append(base + link[0].attrs["href"])
    print("Finished loading ello, found {} urls".format(len(urls)))

    return urls[1:]

def get_ello_urls_listening(url, base):
    doc = load_doc(url)
    sections = doc.find_all("div", attrs={"class": "spacertop"})
    urls = []
    for section in sections:
        groups = section.find_all("div", attrs={"class": "group5"})
        if len(groups) > 1:
            for group in groups:
                link = group.find_all("a")
                if len(link) > 1:
                    urls.append(link[0].attrs["href"].replace("../", base))
    print("Finished loading ello, found {} urls".format(len(urls)))
    return urls

def get_ello_sentences(url, base, is_listening):
    if is_listening:
        urls = get_ello_urls_listening(url, base)
    else:
        urls = get_ello_urls(url, base)
    # urls =["https://elllo.org/english/grammar/L1-12-NatTodd-Adjectives.htm"]
    questions = []
    answers = []

    print("Starting to extract questions and answers from urls...")
    for url in urls:
        senteces = []

        doc = load_doc(url)
        section = doc.find_all("div", attrs={"class": "transcript"})
        if len(section) == 1:
            ps = section[0].find_all("p")
            for paragraph in ps:
                for child in paragraph.children:
                    # if is sentence add to question or answer
                    if type(child) == NavigableString:
                        senteces.append(str(child)[2:])
            q = []
            a = []
            for i in range(len(senteces) - 1):
                q.append(senteces[i])
                a.append(senteces[i + 1])
            print("finished extracting questions and answers from {}, found {} questions and answers".format(doc.title,
                                                                                                             len(q)))
            questions += q
            answers += a

    print("Found {} questions and answers\n".format(len(questions)))
    return questions, answers


print(get_ello_urls_listening("https://elllo.org/english/levels/level3-beginners-high.htm", "https://elllo.org/english/"))

if __name__ == "__main__":
    questions = []
    ansers = []

    # load basic english data
    url = "https://basicenglishspeaking.com/daily-english-conversation-topics/"
    q, a = get_basic_english_converation_data(url)
    questions += q
    ansers += a

    # load elfast data
    q, a = get_elsfast_data("https://www.eslfast.com/easydialogs", "https://www.eslfast.com/easydialogs/index.html",
                            "https://www.eslfast.com")
    questions += q
    ansers += a

    # load ello data
    ello_urls = ["https://elllo.org/english/grammar/index-level1.htm",
                 "https://elllo.org/english/grammar/index-level2.htm",
                 "https://elllo.org/english/grammar/index-level3.htm"]
    for url in ello_urls:
        q, a = get_ello_sentences(url, "https://elllo.org/english/grammar/", False)
        questions += q
        ansers += a

    ello_urls = ["https://elllo.org/english/levels/level3-beginners-high.htm"]
    for url in ello_urls:
        q, a = get_ello_sentences(url, "https://elllo.org/english/", True)
        questions += q
        ansers += a
    import numpy as np

    questions = np.array(questions)
    answers = np.array(ansers)
    np.save("questions.npy", questions)
    np.save("answers.npy", answers)

    questions = np.load("questions.npy")
    answers = np.load("answers.npy")
    idx = np.random.randint(0, questions.shape[0], 10)
    for i in idx:
        q = questions[i]
        a = answers[i]
        print(q)
        print(a)
        print("\n\n")
