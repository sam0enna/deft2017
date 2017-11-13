def load_emotaix(emotaix_file):
    import xml.etree.ElementTree as ET
    emotaix = {}
    current_entry = {}
    current_lemma = None

    for event, elem in ET.iterparse(emotaix_file, events=['start', 'end']):
        if event == 'start':
            if elem.tag == 'Entry':
                current_entry = {}
                current_lemma = None
        elif event == 'end':
            if elem.tag == 'Entry':
                emotaix[current_lemma] = current_entry
            elif elem.tag == 'lemma':
                current_lemma = elem.text
            elif elem.tag == 'pos':
                current_entry[elem.tag] = elem.text
            elif elem.tag == 'sense':
                current_entry[elem.tag] = elem.attrib

    return emotaix


"""
def load_emotaix(emotaix_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(emotaix_file)
    root = tree.getroot()
    for entry in root:

"""


def load_polarimots(polarimots_file):
    """
        family_id : cf. http://polymots.lif.univ-mrs.fr/v2/
        lemma : word lemma
        pos : part of speech(uppercase) + gender(lowercase)
        pol : polarity in ['NEUTRE', 'NEG', 'POS']
        trust : confidence in annotation
    """
    import csv
    header = ['family_id', 'lemma', 'pos', 'pol', 'trust']
    polarimots = {}
    with open(polarimots_file, 'r') as file:
        reader = csv.DictReader(file, fieldnames=header, delimiter='\t', quotechar='"')
        for line in reader:
            if len(line) == 6:
                print(line)
            line = dict(line)
            line['family_id'] = int(line['family_id'])
            line['trust'] = float(line['trust'].replace('%', '').replace(',', '.'))
            lemma = line['lemma']
            line.pop('lemma')
            polarimots[lemma] = line
    return polarimots
