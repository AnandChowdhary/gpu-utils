"""Word lists for the synthetic email generator. Nothing here is copied from a corpus;
names are common given/family names per locale, companies are invented."""

from __future__ import annotations

FIRST_NAMES: dict[str, list[str]] = {
    "en": [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "David",
        "Elizabeth", "William", "Susan", "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen",
        "Daniel", "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Sandra", "Steven", "Ashley",
        "Andrew", "Emily", "Paul", "Donna", "Joshua", "Michelle", "Kevin", "Carol", "Brian", "Amanda",
        "George", "Melissa", "Timothy", "Deborah", "Ryan", "Stephanie", "Jacob", "Rebecca", "Nicholas",
        "Laura", "Eric", "Helen", "Priya", "Aisha", "Wei", "Omar", "Fatima", "Yuki", "Chloe", "Liam",
        "Noah", "Olivia", "Ava", "Ethan", "Mia", "Aiden", "Zoe", "Raj", "Anita", "Tariq", "Sofia",
        "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Jamie", "Riley", "Abhishek", "Deepak",
        "Ananya", "Kavya", "Rohan", "Nikhil", "Sanjay", "Neha", "Vikram", "Pooja", "Arjun", "Meera",
    ],
    "fr": [
        "Jean", "Marie", "Pierre", "Sophie", "Luc", "Camille", "Nicolas", "Julie", "Antoine", "Claire",
        "Julien", "Émilie", "Thomas", "Léa", "Mathieu", "Manon", "Alexandre", "Chloé", "Guillaume",
        "Pauline", "Baptiste", "Élodie", "Romain", "Aurélie", "Hugo", "Inès", "Louis", "Margaux",
        "François", "Hélène", "Étienne", "Céline", "Olivier", "Nathalie",
    ],
    "de": [
        "Hans", "Anna", "Peter", "Julia", "Michael", "Laura", "Thomas", "Sabine", "Andreas", "Katrin",
        "Stefan", "Petra", "Markus", "Claudia", "Christian", "Nicole", "Jan", "Lena", "Lukas", "Sophie",
        "Felix", "Hannah", "Jonas", "Lea", "Tobias", "Katharina", "Florian", "Franziska", "Sebastian",
        "Melanie", "Jürgen", "Ute", "Wolfgang", "Renate", "Matthias", "Birgit",
    ],
    "es": [
        "Carlos", "María", "José", "Ana", "Juan", "Carmen", "Luis", "Laura", "Miguel", "Isabel",
        "Javier", "Lucía", "Antonio", "Marta", "Francisco", "Paula", "Manuel", "Sara", "Diego", "Elena",
        "Pablo", "Andrea", "Alejandro", "Cristina", "Sergio", "Patricia", "Jorge", "Rocío", "Raúl",
        "Beatriz", "Álvaro", "Natalia", "Rafael", "Silvia",
    ],
    "it": [
        "Marco", "Giulia", "Luca", "Francesca", "Andrea", "Chiara", "Matteo", "Sara", "Alessandro",
        "Valentina", "Davide", "Elena", "Giuseppe", "Martina", "Francesco", "Alessia", "Lorenzo",
        "Federica", "Simone", "Silvia", "Giovanni", "Laura", "Paolo", "Elisa",
    ],
    "nl": [
        "Jan", "Anna", "Pieter", "Sanne", "Daan", "Lotte", "Bram", "Emma", "Sem", "Julia", "Lucas",
        "Fleur", "Thijs", "Eva", "Joris", "Lisa", "Ruben", "Sophie", "Bas", "Iris", "Kees", "Maartje",
    ],
    "pt": [
        "João", "Maria", "Pedro", "Ana", "Tiago", "Beatriz", "Miguel", "Inês", "Rui", "Sofia", "Bruno",
        "Mariana", "André", "Carolina", "Diogo", "Rita", "Gabriel", "Juliana", "Rafael", "Larissa",
    ],
    "sv": [
        "Erik", "Anna", "Lars", "Maria", "Karl", "Eva", "Anders", "Kristina", "Johan", "Sara", "Per",
        "Lena", "Nils", "Emma", "Oskar", "Elin", "Magnus", "Karin", "Fredrik", "Sofia",
    ],
    "pl": [
        "Jan", "Anna", "Piotr", "Katarzyna", "Krzysztof", "Małgorzata", "Andrzej", "Agnieszka", "Tomasz",
        "Barbara", "Paweł", "Ewa", "Marcin", "Magdalena", "Michał", "Joanna", "Marek", "Monika",
    ],
    "ja": [
        "Yuki", "Haruto", "Hina", "Sota", "Yui", "Ren", "Aoi", "Yuto", "Sakura", "Kenji", "Takashi",
        "Hiroshi", "Naoko", "Akiko", "Daisuke", "Kaori", "Satoshi", "Mika", "Shun", "Rika",
    ],
    "zh": [
        "Wei", "Fang", "Li", "Jing", "Min", "Jun", "Yan", "Hui", "Lei", "Xin", "Yang", "Mei", "Hao",
        "Ling", "Tao", "Ying", "Qiang", "Na", "Bo", "Xiu",
    ],
    "ar": [
        "Ahmed", "Fatima", "Mohammed", "Aisha", "Ali", "Layla", "Omar", "Noor", "Youssef", "Sara",
        "Hassan", "Mariam", "Khalid", "Huda", "Tariq", "Zainab", "Karim", "Amira", "Samir", "Rania",
    ],
}

LAST_NAMES: dict[str, list[str]] = {
    "en": [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
        "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee",
        "Thompson", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
        "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Hill", "Campbell", "Mitchell", "Roberts",
        "Carter", "Phillips", "Evans", "Turner", "Parker", "Collins", "Edwards", "Stewart", "Morris",
        "Murphy", "Cook", "Rogers", "Patel", "Shah", "Khan", "Singh", "Kumar", "Sharma", "Gupta", "Mehta",
        "Nakamura", "Kim", "Park", "Nguyen", "Tran", "Chen", "Wang", "Zhang", "O'Brien", "McCarthy",
        "Fitzgerald", "Kona", "Reddy", "Iyer", "Nair", "Bose", "Chowdhary", "Malhotra", "Kapoor",
    ],
    "fr": [
        "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy",
        "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux",
        "Vincent", "Fournier", "Morel", "Girard", "André", "Lefèvre", "Mercier", "Dupont", "Lambert",
        "Bonnet", "François", "Martinez", "Legrand", "Garnier", "Faure", "Rousseau",
    ],
    "de": [
        "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz",
        "Hoffmann", "Schäfer", "Koch", "Bauer", "Richter", "Klein", "Wolf", "Schröder", "Neumann",
        "Schwarz", "Zimmermann", "Braun", "Krüger", "Hofmann", "Hartmann", "Lange", "Schmitt", "Werner",
        "Krause", "Meier", "Lehmann", "Huber", "Mayer", "Herrmann", "König",
    ],
    "es": [
        "García", "Fernández", "González", "Rodríguez", "López", "Martínez", "Sánchez", "Pérez", "Gómez",
        "Martín", "Jiménez", "Ruiz", "Hernández", "Díaz", "Moreno", "Muñoz", "Álvarez", "Romero",
        "Alonso", "Gutiérrez", "Navarro", "Torres", "Domínguez", "Vázquez", "Ramos", "Gil", "Ramírez",
        "Serrano", "Blanco", "Molina", "Morales", "Suárez", "Ortega", "Delgado",
    ],
    "it": [
        "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci", "Marino",
        "Greco", "Bruno", "Gallo", "Conti", "De Luca", "Costa", "Giordano", "Mancini", "Rizzo",
        "Lombardi", "Moretti", "Barbieri", "Fontana", "Santoro", "Mariani",
    ],
    "nl": [
        "de Jong", "Jansen", "de Vries", "van den Berg", "van Dijk", "Bakker", "Janssen", "Visser",
        "Smit", "Meijer", "de Boer", "Mulder", "de Groot", "Bos", "Vos", "Peters", "Hendriks",
        "van Leeuwen", "Dekker", "Brouwer", "de Wit", "Dijkstra",
    ],
    "pt": [
        "Silva", "Santos", "Ferreira", "Pereira", "Oliveira", "Costa", "Rodrigues", "Martins", "Jesus",
        "Sousa", "Fernandes", "Gonçalves", "Gomes", "Lopes", "Marques", "Alves", "Almeida", "Ribeiro",
        "Pinto", "Carvalho",
    ],
    "sv": [
        "Andersson", "Johansson", "Karlsson", "Nilsson", "Eriksson", "Larsson", "Olsson", "Persson",
        "Svensson", "Gustafsson", "Pettersson", "Jonsson", "Jansson", "Hansson", "Bengtsson", "Lindberg",
        "Lindqvist", "Lindgren", "Berg", "Axelsson",
    ],
    "pl": [
        "Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kowalczyk", "Kamiński", "Lewandowski", "Zieliński",
        "Szymański", "Woźniak", "Dąbrowski", "Kozłowski", "Jankowski", "Mazur", "Wojciechowski",
        "Kwiatkowski", "Krawczyk", "Kaczmarek",
    ],
    "ja": [
        "Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Ito", "Yamamoto", "Nakamura", "Kobayashi",
        "Kato", "Yoshida", "Yamada", "Sasaki", "Yamaguchi", "Matsumoto", "Inoue", "Kimura", "Hayashi",
        "Shimizu", "Saito",
    ],
    "zh": [
        "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou", "Xu", "Sun", "Ma",
        "Zhu", "Hu", "Guo", "He", "Lin", "Gao", "Luo",
    ],
    "ar": [
        "Al-Sayed", "Hassan", "Ibrahim", "Khalil", "Mansour", "Haddad", "Nasser", "Saleh", "Farah",
        "Abdullah", "Rahman", "Aziz", "Karim", "Hamdan", "Sultan", "Yousef", "Najjar", "Khoury",
    ],
}

# Native-script names used for JA/ZH signatures and attribution lines.
JA_NAMES = [
    "佐藤 健", "鈴木 花子", "高橋 一郎", "田中 美咲", "渡辺 翔太", "伊藤 由美", "山本 太郎", "中村 さくら",
    "小林 大輔", "加藤 直子", "吉田 拓也", "山田 優子", "佐々木 誠", "松本 恵", "井上 隆", "木村 愛",
]
ZH_NAMES = [
    "王伟", "李娜", "张强", "刘洋", "陈静", "杨勇", "黄敏", "赵磊", "吴芳", "周杰", "徐丽", "孙浩",
    "马超", "朱婷", "胡军", "郭燕", "何平", "林峰", "高翔", "罗琳",
]

TITLES: list[str] = [
    "CEO", "CTO", "CFO", "COO", "Founder", "Co-Founder", "Founder & CEO", "President",
    "Vice President", "VP of Engineering", "VP, Sales", "VP Marketing", "Managing Director",
    "Director", "Director of Operations", "Director, Product Management", "Head of Product",
    "Head of Design", "Head of Growth", "Chief of Staff", "General Manager", "Regional Manager",
    "Account Manager", "Account Executive", "Senior Account Executive", "Sales Manager",
    "Sales Development Representative", "Business Development Manager", "Customer Success Manager",
    "Customer Support Specialist", "Support Engineer", "Software Engineer", "Senior Software Engineer",
    "Staff Engineer", "Principal Engineer", "Engineering Manager", "Frontend Developer",
    "Backend Developer", "Full Stack Developer", "iOS Developer", "Data Scientist", "Data Analyst",
    "Machine Learning Engineer", "DevOps Engineer", "Site Reliability Engineer", "QA Engineer",
    "Product Manager", "Senior Product Manager", "Product Designer", "UX Designer", "UX Researcher",
    "Graphic Designer", "Art Director", "Creative Director", "Marketing Manager",
    "Content Marketing Manager", "Marketing Coordinator", "Communications Manager", "PR Manager",
    "Social Media Manager", "Community Manager", "Recruiter", "Technical Recruiter", "HR Manager",
    "HR Business Partner", "People Operations", "Office Manager", "Executive Assistant",
    "Administrative Assistant", "Project Manager", "Program Manager", "Scrum Master",
    "Operations Manager", "Supply Chain Manager", "Procurement Specialist", "Logistics Coordinator",
    "Finance Manager", "Controller", "Accountant", "Senior Accountant", "Financial Analyst",
    "Legal Counsel", "General Counsel", "Paralegal", "Attorney at Law", "Partner", "Associate",
    "Consultant", "Senior Consultant", "Principal Consultant", "Analyst", "Research Scientist",
    "Professor", "Associate Professor", "Assistant Professor", "Lecturer", "PhD Candidate",
    "Research Assistant", "Postdoctoral Researcher", "Teacher", "Principal", "Nurse Practitioner",
    "Physician", "Pharmacist", "Architect", "Civil Engineer", "Mechanical Engineer",
    "Electrical Engineer", "Real Estate Agent", "Broker", "Insurance Agent", "Journalist", "Editor",
    "Photographer", "Owner", "Proprietor", "Freelance Writer", "Independent Consultant",
    "Solutions Architect", "Cloud Architect", "Security Analyst", "IT Manager", "Systems Administrator",
    "Network Engineer", "Technical Writer", "Developer Advocate", "Developer Relations",
    "Partnerships Lead", "Growth Lead", "Chief Marketing Officer", "Chief Product Officer",
    "Chief Revenue Officer", "Chief Information Officer", "Country Manager", "Team Lead",
    "Tech Lead", "Store Manager", "Branch Manager", "Event Coordinator", "Travel Consultant",
    # non-English
    "Directeur Général", "Directrice Marketing", "Responsable Commercial", "Chef de Projet",
    "Ingénieur Logiciel", "Chargée de Communication", "Geschäftsführer", "Geschäftsführerin",
    "Leiter Vertrieb", "Projektleiterin", "Softwareentwickler", "Kundenberater", "Prokurist",
    "Director General", "Gerente de Ventas", "Jefe de Proyecto", "Ingeniera de Software",
    "Responsable de Marketing", "Direttore Commerciale", "Responsabile Marketing", "Ingegnere",
    "営業部長", "代表取締役", "プロジェクトマネージャー", "総经理", "销售经理", "项目经理", "市场总监",
]

COMPANY_STEMS: list[str] = [
    "Acme", "Northwind", "Globex", "Initech", "Umbrella", "Vandelay", "Stark", "Wayne", "Hooli",
    "Pied Piper", "Aperture", "Cyberdyne", "Tyrell", "Wonka", "Dunder Mifflin", "Sterling Cooper",
    "Bluth", "Oceanic", "Massive Dynamic", "Soylent", "Gringotts", "Nakatomi", "Weyland", "Zorg",
    "Lumon", "Prestige", "Vector", "Summit", "Harbor", "Meridian", "Pinnacle", "Beacon", "Atlas",
    "Orion", "Nova", "Vertex", "Quantum", "Apex", "Horizon", "Evergreen", "Silverline", "Bluebird",
    "Redwood", "Ironclad", "Brightpath", "Clearwater", "Crestview", "Fairmont", "Greenfield",
    "Lakeshore", "Maple", "Oakridge", "Riverbend", "Stonebridge", "Sunrise", "Westfield", "Cobalt",
    "Nimbus", "Lattice", "Kestrel", "Halcyon", "Solstice", "Tessera", "Fathom", "Zenith", "Lumen",
    "Arbor", "Cinder", "Ember", "Fjord", "Glacier", "Juniper", "Marlow", "Onyx", "Quill", "Sable",
    "Tundra", "Vale", "Willow", "Yarrow", "Bright", "Swift", "Keystone", "Cornerstone", "Anchor",
    "Pilot", "Compass", "Vantage", "Sentinel", "Trident", "Aegis", "Helix", "Prism", "Spectrum",
    "Mosaic", "Canvas", "Foundry", "Forge", "Loom", "Kiln", "Mill", "Grove", "Orchard", "Meadow",
    "Bergmann", "Dupont", "Schneider", "Rossi", "Tanaka", "Lindqvist", "Patel", "Novak",
]
COMPANY_KINDS: list[str] = [
    "Technologies", "Software", "Labs", "Systems", "Solutions", "Consulting", "Partners", "Group",
    "Holdings", "Capital", "Ventures", "Media", "Studio", "Studios", "Design", "Digital", "Analytics",
    "Logistics", "Industries", "Manufacturing", "Foods", "Health", "Healthcare", "Medical", "Legal",
    "Law", "Realty", "Properties", "Travel", "Energy", "Robotics", "Networks", "Cloud", "Security",
    "Insurance", "Bank", "Financial", "Pharma", "Biotech", "Motors", "Marine", "Aerospace",
    "Education", "Publishing", "Books", "Coffee", "Bakery", "Brewing", "Apparel", "Outdoors",
    "Fitness", "Interactive", "Games", "Music", "Films", "Events", "Marketing", "Creative",
    "Automation", "Semiconductors", "Materials", "Packaging", "Print", "Textiles", "Furniture",
]
COMPANY_SUFFIXES: list[str] = [
    "Inc", "Inc.", "LLC", "Ltd", "Ltd.", "Limited", "Corp", "Corp.", "Corporation", "Co.", "Co",
    "PLC", "GmbH", "AG", "KG", "GmbH & Co. KG", "SAS", "SARL", "SA", "S.A.", "S.L.", "S.r.l.",
    "S.p.A.", "B.V.", "N.V.", "AB", "Oy", "Pty Ltd", "Pvt. Ltd.", "Pvt Ltd", "LLP", "LP", "Sp. z o.o.",
    "株式会社", "有限公司", "Ltda.", "S.A. de C.V.",
]
COMPANY_EXTRA: list[str] = [
    "The {stem} Company", "{stem}.io", "{stem}.ai", "{stem}.com", "{stem} & Sons", "{stem} & Co.",
    "{stem} {kind} International", "{stem} {kind} Europe", "{stem} {kind} Asia Pacific",
    "University of {stem}", "{stem} University", "{stem} Institute", "{stem} Foundation",
    "{stem} Hospital", "{stem} Clinic", "{stem} School", "{stem} College", "City of {stem}",
    "{stem} County", "{stem} Bank", "{stem} Credit Union",
]

DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Finance", "Legal", "Operations", "Product", "Design",
    "Customer Success", "Support", "Human Resources", "People", "Research", "IT", "Security",
    "Procurement", "Communications", "Partnerships", "Data", "Platform", "Infrastructure",
]

STREET_NAMES = [
    "Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Lake", "Hill", "Park", "River",
    "Church", "Market", "High", "Station", "Bridge", "King", "Queen", "Victoria", "Albert", "Mill",
    "Spring", "Union", "Broad", "Center", "Sunset", "Lincoln", "Jefferson", "Madison", "Franklin",
    "Harrison", "Jackson", "Adams", "Monroe", "Grand", "Ocean", "Bay", "Forest", "Meadow", "Valley",
]
STREET_TYPES = ["St", "St.", "Street", "Ave", "Ave.", "Avenue", "Rd", "Road", "Blvd", "Boulevard",
                "Dr", "Drive", "Ln", "Lane", "Way", "Ct", "Court", "Pl", "Place", "Pkwy", "Parkway"]
US_CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Houston", "TX"), ("Phoenix", "AZ"),
    ("Philadelphia", "PA"), ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("San Jose", "CA"), ("Austin", "TX"), ("Seattle", "WA"), ("Denver", "CO"), ("Boston", "MA"),
    ("Portland", "OR"), ("Atlanta", "GA"), ("Miami", "FL"), ("Minneapolis", "MN"), ("Nashville", "TN"),
    ("Springfield", "IL"), ("Raleigh", "NC"), ("Salt Lake City", "UT"), ("Pittsburgh", "PA"),
    ("San Francisco", "CA"), ("Oakland", "CA"), ("Brooklyn", "NY"), ("Cambridge", "MA"),
]
UK_CITIES = ["London", "Manchester", "Birmingham", "Leeds", "Bristol", "Edinburgh", "Glasgow",
             "Cambridge", "Oxford", "Reading", "Brighton", "Cardiff", "Belfast", "Liverpool"]
UK_POSTCODES = ["SW1A 1AA", "EC2A 4NE", "M1 1AE", "B1 1AA", "LS1 4AP", "BS1 4DJ", "EH1 1YZ", "G1 1XQ",
                "CB2 1TN", "OX1 2JD", "RG1 1AX", "BN1 1AA", "CF10 1EP", "BT1 5GS", "L1 8JQ", "N1 9GU",
                "W1D 3QF", "SE1 7PB", "E1 6AN", "NW1 2DB"]
DE_CITIES = ["Berlin", "München", "Hamburg", "Köln", "Frankfurt am Main", "Stuttgart", "Düsseldorf",
             "Leipzig", "Dresden", "Hannover", "Nürnberg", "Bremen", "Wien", "Zürich", "Bonn"]
DE_STREETS = ["Hauptstraße", "Bahnhofstraße", "Schillerstraße", "Goethestraße", "Gartenstraße",
              "Berliner Straße", "Kirchweg", "Lindenallee", "Am Markt", "Friedrichstraße",
              "Marienplatz", "Königsallee", "Rosenweg", "Mühlenweg", "Waldstraße"]
FR_CITIES = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Bordeaux",
             "Lille", "Rennes", "Montpellier", "Grenoble", "Genève", "Bruxelles", "Lausanne"]
FR_STREETS = ["rue de la Paix", "avenue des Champs-Élysées", "boulevard Saint-Germain",
              "rue de Rivoli", "rue Victor Hugo", "place de la République", "avenue Jean Jaurès",
              "rue du Commerce", "chemin des Vignes", "allée des Tilleuls", "cours Lafayette",
              "rue Nationale", "quai des Bergues"]
ES_CITIES = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao", "Málaga", "Zaragoza",
             "Ciudad de México", "Bogotá", "Buenos Aires", "Lima", "Santiago"]
ES_STREETS = ["Calle Mayor", "Calle de Alcalá", "Gran Vía", "Paseo de la Castellana", "Avenida Diagonal",
              "Calle Serrano", "Plaza Mayor", "Carrer de Balmes", "Avenida Insurgentes", "Calle 72"]
IN_CITIES = [("Mumbai", "Maharashtra", "400001"), ("Bengaluru", "Karnataka", "560001"),
             ("New Delhi", "Delhi", "110001"), ("Hyderabad", "Telangana", "500001"),
             ("Pune", "Maharashtra", "411001"), ("Chennai", "Tamil Nadu", "600001"),
             ("Gurugram", "Haryana", "122001"), ("Noida", "Uttar Pradesh", "201301"),
             ("Kolkata", "West Bengal", "700001"), ("Ahmedabad", "Gujarat", "380001")]
IN_STREETS = ["MG Road", "Brigade Road", "Nehru Place", "Bandra Kurla Complex", "Hitech City",
              "Koramangala 4th Block", "Sector 44", "Anna Salai", "Park Street", "SG Highway",
              "Whitefield Main Road", "Andheri East", "Connaught Place", "Cyber City"]
AU_CITIES = [("Sydney", "NSW", "2000"), ("Melbourne", "VIC", "3000"), ("Brisbane", "QLD", "4000"),
             ("Perth", "WA", "6000"), ("Adelaide", "SA", "5000"), ("Canberra", "ACT", "2600")]
CA_CITIES = [("Toronto", "ON", "M5V 2T6"), ("Vancouver", "BC", "V6B 1A1"), ("Montréal", "QC", "H3B 2Y5"),
             ("Calgary", "AB", "T2P 1J9"), ("Ottawa", "ON", "K1P 5G3")]
JA_ADDRESSES = ["東京都千代田区丸の内1-1-1", "東京都港区六本木6-10-1", "大阪府大阪市北区梅田3-1-3",
                "神奈川県横浜市西区みなとみらい2-2-1", "愛知県名古屋市中村区名駅1-1-4",
                "〒100-0005 東京都千代田区丸の内2-4-1", "〒530-0001 大阪市北区梅田1-1-3"]
ZH_ADDRESSES = ["北京市朝阳区建国路88号", "上海市浦东新区世纪大道100号", "深圳市南山区科技园南区",
                "广州市天河区珠江新城华夏路10号", "杭州市西湖区文三路90号", "北京市海淀区中关村大街1号"]

EMAIL_DOMAINS_PERSONAL = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
                          "me.com", "protonmail.com", "gmx.de", "web.de", "orange.fr", "free.fr",
                          "yahoo.co.jp", "163.com", "qq.com", "mail.ru", "live.com", "aol.com",
                          "hey.com", "fastmail.com", "posteo.de", "laposte.net", "t-online.de"]
TLDS = ["com", "com", "com", "io", "co", "net", "org", "ai", "dev", "co.uk", "de", "fr", "es", "it",
        "nl", "se", "jp", "cn", "in", "com.au", "ca", "ch", "at", "be", "eu", "app", "tech", "us"]

GREETINGS: dict[str, list[str]] = {
    "en": [
        "Hi {first},", "Hi {first}", "Hello {first},", "Hey {first},", "Hey {first}!", "Dear {first},",
        "Dear {title_last},", "Dear {full},", "Hi,", "Hello,", "Hi there,", "Hi all,", "Hi team,",
        "Hello everyone,", "Hey folks,", "Hi both,", "Good morning {first},", "Good afternoon,",
        "Good morning,", "Morning {first},", "Dear Sir or Madam,", "Dear Hiring Manager,",
        "Dear Team,", "To whom it may concern,", "Hi {first} and {first2},", "{first},", "{first} -",
        "Hey,", "Hiya {first},", "Hello {full},", "Dear Dr. {last},", "Greetings,", "Hi {first}:",
        "Hello {first}:", "Dear {first}:", "Hi everyone",
    ],
    "fr": ["Bonjour {first},", "Bonjour,", "Bonjour {title_last},", "Salut {first},", "Cher {first},",
           "Chère {first},", "Bonjour à tous,", "Madame, Monsieur,", "Bonsoir {first},", "Hello {first},"],
    "de": ["Hallo {first},", "Hallo,", "Hi {first},", "Liebe {first},", "Lieber {first},",
           "Sehr geehrte Frau {last},", "Sehr geehrter Herr {last},", "Sehr geehrte Damen und Herren,",
           "Guten Tag {title_last},", "Guten Morgen {first},", "Hallo zusammen,", "Moin {first},",
           "Servus {first},"],
    "es": ["Hola {first},", "Hola,", "Buenos días {first},", "Buenas tardes,", "Estimado {first},",
           "Estimada {first},", "Estimado Sr. {last},", "Estimada Sra. {last},", "Hola a todos,",
           "Querido {first},", "Buenas,"],
    "it": ["Ciao {first},", "Buongiorno {first},", "Gentile {full},", "Salve,", "Buongiorno,",
           "Ciao a tutti,"],
    "nl": ["Hoi {first},", "Beste {first},", "Hallo {first},", "Geachte heer {last},",
           "Geachte mevrouw {last},", "Dag {first},"],
    "pt": ["Olá {first},", "Oi {first},", "Bom dia {first},", "Prezado {first},", "Prezada {first},",
           "Caro {first},"],
    "sv": ["Hej {first},", "Hej,", "Hej {first}!", "Hejsan,"],
    "pl": ["Cześć {first},", "Dzień dobry,", "Szanowny Panie {last},", "Witam,"],
    "ja": ["{last_ja}様", "{last_ja}さん", "{full_ja} 様", "お世話になっております。", "お疲れ様です。",
           "{last_ja}様、お世話になっております。", "{first}さん、こんにちは"],
    "zh": ["{last_zh}总，您好：", "{full_zh}，你好：", "您好，", "你好，", "各位好，", "Hi {first},",
           "{full_zh}您好，"],
    "ar": ["Dear {first},", "Hi {first},", "Hello {first},", "Assalamu alaikum {first},"],
}

CLOSINGS: dict[str, list[str]] = {
    "en": [
        "Best,", "Best regards,", "Kind regards,", "Regards,", "Warm regards,", "Thanks,", "Thanks!",
        "Thank you,", "Thank you!", "Thanks again,", "Many thanks,", "Cheers,", "Cheers!", "Best wishes,",
        "All the best,", "Sincerely,", "Sincerely yours,", "Yours truly,", "Yours sincerely,",
        "Talk soon,", "Speak soon,", "Take care,", "Warmly,", "Best", "Thanks", "Cheers", "Regards",
        "Thx,", "Thanks in advance,", "Thanks so much,", "Much appreciated,", "With gratitude,",
        "Respectfully,", "Have a great weekend,", "Have a good one,", "Looking forward to it,",
        "Let me know if you have any questions.", "Hope this helps,", "Best regards", "Kind regards",
        "Thanks a lot,", "Warm wishes,", "Rgds,", "BR,", "-", "--", "Ta,", "Peace,", "Onwards,",
    ],
    "fr": ["Cordialement,", "Bien cordialement,", "Bien à vous,", "Merci,", "Merci d'avance,",
           "Amicalement,", "Bonne journée,", "Sincères salutations,", "Cordialement", "À bientôt,",
           "Bises,", "Meilleures salutations,", "Merci beaucoup,"],
    "de": ["Mit freundlichen Grüßen", "Mit freundlichen Grüßen,", "Viele Grüße", "Viele Grüße,",
           "Beste Grüße", "Liebe Grüße", "Freundliche Grüße", "Schöne Grüße", "Herzliche Grüße",
           "Danke und Gruß", "Gruß", "LG", "VG", "MfG", "Vielen Dank!", "Vielen Dank im Voraus,",
           "Bis bald,"],
    "es": ["Saludos,", "Un saludo,", "Saludos cordiales,", "Atentamente,", "Gracias,",
           "Muchas gracias,", "Un abrazo,", "Cordialmente,", "Gracias de antemano,", "Hasta pronto,"],
    "it": ["Cordiali saluti,", "Saluti,", "Grazie,", "Grazie mille,", "A presto,", "Buona giornata,",
           "Distinti saluti,"],
    "nl": ["Met vriendelijke groet,", "Groeten,", "Hartelijke groet,", "Bedankt,", "Groetjes,", "Mvg,"],
    "pt": ["Atenciosamente,", "Abraços,", "Obrigado,", "Obrigada,", "Cumprimentos,", "Um abraço,"],
    "sv": ["Med vänliga hälsningar,", "Vänliga hälsningar,", "Mvh,", "Tack,", "Hälsningar,"],
    "pl": ["Pozdrawiam,", "Z poważaniem,", "Dziękuję,", "Pozdrawiam serdecznie,"],
    "ja": ["よろしくお願いいたします。", "よろしくお願いします。", "以上、よろしくお願いいたします。",
           "何卒よろしくお願い申し上げます。", "引き続きよろしくお願いいたします。", "ありがとうございます。"],
    "zh": ["谢谢！", "谢谢", "祝好，", "此致", "敬礼", "顺祝商祺", "祝工作顺利", "谢谢您的帮助。", "Best,"],
    "ar": ["Best regards,", "Regards,", "Thanks,", "Kind regards,", "Shukran,"],
}

MOBILE_SIGS: list[str] = [
    "Sent from my iPhone", "Sent from my iPad", "Sent from my Android", "Sent from my Samsung Galaxy",
    "Sent from my Galaxy", "Sent from my BlackBerry", "Sent from my Verizon Wireless BlackBerry",
    "Sent from my Pixel", "Sent from my Google Pixel", "Sent from my Huawei phone", "Sent from my phone",
    "Sent from my mobile", "Sent from my mobile device", "Sent from my Samsung device",
    "Sent from my T-Mobile 4G LTE Device", "Sent from my Verizon 4G LTE smartphone",
    "Sent from my iPhone using Mail", "Sent from Mail for Windows", "Sent from Mail for Windows 10",
    "Sent from Outlook for iOS", "Sent from Outlook for Android", "Get Outlook for iOS",
    "Get Outlook for Android", "Sent from Yahoo Mail on Android", "Sent from Yahoo Mail for iPhone",
    "Sent from Gmail Mobile", "Sent via Superhuman", "Sent via Superhuman iOS", "Sent with Spark",
    "Sent with Proton Mail secure email.", "Sent from Front", "Sent from a mobile device, please excuse typos.",
    "Sent from my iPhone, please excuse any typos", "Sent from my phone - apologies for brevity",
    "Sent from a magnificent torch of pixels", "Sent from my Windows Phone", "Sent from my LG phone",
    "Sent from my Xiaomi", "Sent from my OnePlus", "Sent from my HTC", "Sent from my Nokia",
    "Envoyé de mon iPhone", "Envoyé de mon iPad", "Envoyé depuis mon smartphone Samsung Galaxy",
    "Envoyé de mon mobile", "Von meinem iPhone gesendet", "Von meinem iPad gesendet",
    "Von meinem Samsung Galaxy Smartphone gesendet", "Von meinem Huawei-Telefon gesendet",
    "Von meinem Mobiltelefon gesendet", "Enviado desde mi iPhone", "Enviado desde mi iPad",
    "Enviado desde mi Samsung Galaxy", "Enviado desde mi teléfono", "Inviato da iPhone",
    "Inviato dal mio smartphone Samsung Galaxy", "Verzonden vanaf mijn iPhone",
    "Verzonden vanaf mijn Samsung Galaxy-smartphone", "Enviado do meu iPhone", "Skickat från min iPhone",
    "Wysłane z iPhone'a", "iPhoneから送信", "iPadから送信", "Galaxyから送信", "Outlook for iOS を入手",
    "iPhoneから送信されました", "从我的 iPhone 发送", "发自我的iPhone", "发自我的 iPhone", "从我的华为手机发送",
    "Sent from my iPhone 15", "Sent from my Samsung Galaxy smartphone.", "Sent from my iPhone.",
]

DISCLAIMERS: list[str] = [
    "This email and any attachments are confidential and intended solely for the use of the individual or entity to whom they are addressed. If you have received this email in error please notify the sender immediately and delete it from your system.",
    "CONFIDENTIALITY NOTICE: This message contains confidential information and is intended only for the individual named. If you are not the named addressee you should not disseminate, distribute or copy this email. Please notify the sender immediately by email if you have received this email by mistake.",
    "The information contained in this communication is intended solely for the use of the individual or entity to whom it is addressed and others authorized to receive it. It may contain confidential or legally privileged information.",
    "This e-mail message, including any attachments, is for the sole use of the intended recipient(s) and may contain confidential and privileged information. Any unauthorized review, use, disclosure or distribution is prohibited.",
    "Please consider the environment before printing this email.",
    "This message has been scanned for viruses and dangerous content and is believed to be clean.",
    "Disclaimer: The views expressed in this email are those of the sender and do not necessarily reflect the views of {company}.",
    "{company} is a company registered in England and Wales with company number {regno}. Registered office: {address}.",
    "{company} | Registered in {country} No. {regno} | VAT No. {vat}",
    "This email has been checked for viruses by Avast antivirus software.",
    "Diese E-Mail enthält vertrauliche und/oder rechtlich geschützte Informationen. Wenn Sie nicht der richtige Adressat sind oder diese E-Mail irrtümlich erhalten haben, informieren Sie bitte sofort den Absender und vernichten Sie diese Mail.",
    "Ce message et toutes les pièces jointes sont confidentiels et établis à l'intention exclusive de ses destinataires. Toute utilisation ou diffusion non autorisée est interdite. Si vous recevez ce message par erreur, merci d'en avertir immédiatement l'expéditeur.",
    "Este mensaje y sus archivos adjuntos son confidenciales y van dirigidos exclusivamente a su destinatario. Si usted no es el destinatario, le rogamos lo comunique al remitente y proceda a su eliminación.",
    "This email is intended only for the person(s) to whom it is addressed and may contain confidential information. Any use, copying or distribution by anyone else is prohibited. The sender accepts no liability for any damage caused by any virus transmitted by this email.",
    "IMPORTANT: The contents of this email and any attachments are confidential. They are intended for the named recipient(s) only. If you have received this email by mistake, please notify the sender immediately and do not disclose the contents to anyone or make copies thereof.",
    "NOTICE: This e-mail transmission, and any documents, files or previous e-mail messages attached to it, may contain confidential information that is legally privileged. Unauthorized use is strictly prohibited.",
    "Save a tree. Don't print this e-mail unless it's really necessary.",
    "You are receiving this email because you signed up for updates from {company}. To unsubscribe, click here: {url}",
    "To unsubscribe from this list, send an email to {email} or visit {url}",
    "{company} will never ask you for your password by email. If you believe you have received a suspicious message, please report it to {email}.",
    "This message is from an external sender. Please be cautious of links and attachments.",
    "[EXTERNAL EMAIL] Do not click links or open attachments unless you recognize the sender and know the content is safe.",
    "Copyright © {year} {company}. All rights reserved.",
    "{company}, {address}",
    "Privileged & Confidential. Attorney-Client Communication / Attorney Work Product.",
    "Nothing in this email shall be construed as legally binding upon {company}.",
    "This communication does not constitute an offer, solicitation or recommendation.",
    "Registered in Ireland No. {regno}. Directors: {name1}, {name2}.",
]
MAILING_LIST_FOOTERS: list[list[str]] = [
    ["_______________________________________________", "{list} mailing list", "{list}@{domain}", "http://lists.{domain}/mailman/listinfo/{list}"],
    ["--", "You received this message because you are subscribed to the Google Groups \"{list}\" group.", "To unsubscribe from this group and stop receiving emails from it, send an email to {list}+unsubscribe@googlegroups.com.", "To view this discussion on the web visit https://groups.google.com/d/msgid/{list}/{id}."],
    ["---", "Reply to this email directly or view it on GitHub:", "https://github.com/{org}/{repo}/issues/{num}#issuecomment-{id}"],
    ["--", "Reply to this email directly, view it on GitHub, or unsubscribe.", "You are receiving this because you were mentioned."],
    ["-- ", "You received this bug notification because you are subscribed to {repo}.", "https://bugs.launchpad.net/bugs/{num}"],
    ["##- Please type your reply above this line -##"],
    ["-----", "This email was sent by {company}. Manage your notification settings at {url}"],
    ["View this email in your browser: {url}", "Unsubscribe | Update preferences"],
]

BODY_SENTENCES: list[str] = [
    "Thanks for getting back to me so quickly.", "I hope you're doing well.", "I hope this finds you well.",
    "Just following up on my previous email.", "Quick question about the {noun}.",
    "Could you send over the latest version of the {noun} when you get a chance?",
    "I've attached the {noun} for your review.", "Let me know if that works for you.",
    "Are you free for a call {day} afternoon?", "We should be able to ship this by {day}.",
    "The {noun} looks good to me, but I have a couple of small comments.",
    "I'm not sure I understand the second point, could you clarify?",
    "Please find the updated {noun} attached.", "Sorry for the delay in responding.",
    "That sounds great, let's do it.", "I'll take a look and get back to you by end of day.",
    "We ran into an issue with the {noun} on {day}.", "Can we move the meeting to {time}?",
    "I've cc'd {name} who is leading the {noun} on our side.", "What's the best way to reach you?",
    "The numbers for Q{q} are in and they look better than expected.", "I don't think that's going to work.",
    "Do you have an ETA for the {noun}?", "This is a gentle reminder that the {noun} is due {day}.",
    "Let's sync on this next week.", "Happy to jump on a call if easier.", "Thanks for flagging this.",
    "I agree with {name} on this one.", "I'd rather we didn't rush this.", "See below for details.",
    "FYI, see the thread below.", "Forwarding this in case you missed it.", "Any thoughts?",
    "Would {day} at {time} work?", "I have a conflict at that time, unfortunately.",
    "Here's the link: {url}", "You can find it at {url}", "The invoice total comes to ${amount}.",
    "We paid ${amount} last month for the same thing.", "Our budget is capped at ${amount} for now.",
    "The build is failing on {os} with the following error:", "I tried restarting but it didn't help.",
    "Steps to reproduce:", "Expected behaviour: the page loads.", "Actual behaviour: a blank screen.",
    "It works on my machine.", "I'll open a ticket.", "Ticket number is {num}.", "Order #{num} has shipped.",
    "Your appointment is confirmed for {day} at {time}.", "Please bring a photo ID.",
    "We're located at {address}.", "Call me at {phone} if anything comes up.", "My cell is {phone}.",
    "You can also email {email} directly.", "The meeting room is booked under my name.",
    "I wrote the first draft over the weekend.", "She wrote back saying it was fine.",
    "He sent it from his phone so it's a bit terse.", "Sent from my desk, is much easier than my phone.",
    "As discussed, I'm sending over the proposal.", "Per my last email, the deadline hasn't changed.",
    "Not sure if this is the right address for this.", "Adding {name} to the thread.",
    "One more thing: the {noun} still needs sign-off from legal.", "That's all from me for now.",
    "Ok.", "Sounds good.", "Great, thanks!", "Perfect.", "Will do.", "Noted.", "+1", ":+1:", "Done.",
    "Yes.", "No.", "Sure.", "Got it, thanks.", "Awesome!", "LGTM", "Ack.", "Thanks for the heads up.",
    "Confirming receipt.", "Received, thank you.", "Approved.", "Please proceed.",
    "Let me know.", "Any update on this?", "Bumping this.", "Any news?", "Ping.",
    "Best case we finish {day}, worst case the week after.", "I'm out of office until {day}.",
    "I'll be travelling next week with limited access to email.", "Can you take this one?",
    "The {noun} was better than the last one, honestly.", "This is the third time this has happened.",
    "Long story short, we need a new {noun}.", "TL;DR: it's fixed.", "Regards to the team!",
    "Thanks to everyone who helped on the {noun}.", "The best part was the demo.",
    "In the end we went with option B.", "Two things: the {noun} and the {noun2}.",
    "1. Finish the {noun}", "2. Review the {noun2}", "3. Ship it", "- {noun}", "- {noun2}",
    "* {noun} (owner: {name})", "* {noun2} (due {day})", "• {noun}", "• {noun2}",
    "The tests pass locally but fail in CI.", "See {url} for the full report.",
    "Attached: {noun}.pdf", "Attachment: {noun}_v{num}.xlsx", "PFA the {noun}.",
    "Kindly revert at the earliest.", "Please do the needful.", "Gentle reminder.",
    "Apologies, I replied to the wrong thread.", "Ignore my last message.", "Scratch that.",
    "Hope you had a good weekend.", "How was the trip?", "Congrats on the launch!",
    "Welcome aboard!", "Nice to meet you yesterday.", "It was great catching up.",
    "I'd love to hear your thoughts on the {noun}.", "Feel free to reach out anytime.",
    "Thanks in advance for your help with the {noun}.", "Much appreciated.",
    "I am currently using the Java HTTP API.", "What is the best way to clear the cache after a test?",
    "The function returns null when the input is empty.", "Can you check the logs from {day}?",
    "Here is the config I'm using:", "    timeout: 30", "    retries: 3", "    host: 127.0.0.1",
    "$ npm run build", "$ git push origin main", "error: something went wrong (code {num})",
    "Right, makes sense.",
    "Also, are we still on for lunch?", "PS: don't forget the {noun}.", "P.S. Say hi to {name}!",
    "PPS: the {noun} is in the shared drive.", "Warm regards to your family.",
    "Best not to mention the {noun} in the meeting.", "Thanks for nothing.", "Kind of a mess, honestly.",
    "Dear god, the {noun} again.", "Hi-fi speakers are not in scope.", "The Hello page is broken.",
    "All the best parts are in the {noun}.", "From what I can tell, it's a config issue.",
    "Subject matter experts disagree.", "Date of the event is TBD.", "To be honest, I forgot.",
    "On the other hand, the {noun} is cheaper.", "On balance I'd go with the {noun}.",
    "On {day} I'll be in the office.", "Sent it over just now.", "I wrote to {name} about it last week.",
    "Original plan was {day}, now it's {day2}.", "The message from {name} was clear.",
    "Le {noun} est prêt.", "Je reviens vers vous {day}.", "Merci pour votre retour rapide.",
    "Pouvez-vous m'envoyer le {noun} ?", "Je vous confirme notre rendez-vous de {time}.",
    "C'est noté, merci.", "Je suis en déplacement toute la semaine.",
    "Das klingt gut.", "Können Sie mir das {noun} bis {day} schicken?", "Vielen Dank für die schnelle Antwort.",
    "Ich melde mich morgen.", "Der Termin passt mir.", "Anbei die aktuelle Version.",
    "Ich bin bis {day} im Urlaub.", "Passt das für Sie?",
    "Me parece bien.", "¿Puedes enviarme el {noun}?", "Gracias por la respuesta.", "Adjunto el documento.",
    "Nos vemos el {day}.", "¿Te viene bien a las {time}?", "Estoy fuera de la oficina hasta el {day}.",
    "Va bene, grazie.", "Ti mando il {noun} domani.", "Ci sentiamo la prossima settimana.",
    "Prima, dank je.", "Kun je het {noun} sturen?", "Tot {day}!",
    "Tudo bem por aqui.", "Envio o {noun} em anexo.", "Obrigado pelo retorno.",
    "Tack för svaret.", "Vi hörs på {day}.", "Dziękuję za odpowiedź.", "Do zobaczenia w {day}.",
    "ご確認のほどよろしくお願いいたします。", "添付ファイルをご確認ください。", "明日の会議は{time}からです。",
    "ご返信ありがとうございます。", "了解しました。", "お手数をおかけしますが、よろしくお願いします。",
    "请查收附件。", "明天{time}开会。", "收到，谢谢。", "请尽快回复。", "这个问题我们下周讨论。", "没问题。",
]
NOUNS = ["report", "proposal", "contract", "invoice", "deck", "spreadsheet", "design", "mockup",
         "budget", "roadmap", "release", "migration", "database", "API", "prototype", "survey",
         "agenda", "quote", "estimate", "schedule", "timeline", "brief", "summary", "draft", "PR",
         "pull request", "ticket", "bug", "feature", "campaign", "newsletter", "onboarding doc",
         "presentation", "pitch", "demo", "test plan", "backlog", "sprint", "retro notes", "OKRs",
         "offer letter", "NDA", "SOW", "MSA", "renewal", "order", "shipment", "sample", "logo",
         "website", "landing page", "app", "dashboard", "pipeline", "model", "dataset", "config",
         "riak bucket", "cache", "loader", "build", "deploy", "cluster", "server", "backup"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "next week", "the 15th",
        "Friday the 3rd", "Mon", "Tue", "Wed", "Thu", "Fri", "EOD", "end of week", "the weekend",
        "lundi", "mardi", "Montag", "Freitag", "lunes", "viernes", "next Monday", "March 3", "Jan 10"]
TIMES = ["3pm", "3 PM", "15:00", "9:30", "9:30am", "noon", "10", "10am", "2:15 PM", "14h", "16 Uhr",
         "las 10", "5", "half past four", "11:00 CET", "8am PST"]
OSES = ["Windows", "macOS", "Ubuntu 22.04", "Linux", "iOS 17", "Android 14", "Windows 11", "Debian"]

SUBJECTS = ["Re: {noun}", "RE: {noun} update", "Fwd: {noun}", "FW: {noun} - action required",
            "{noun} for {day}", "Question about the {noun}", "[{company}] {noun}", "Re: Re: {noun}",
            "AW: {noun}", "TR: {noun}", "RV: {noun}", "Meeting {day}", "Invoice #{num}", "Hello",
            "Test", "Loop details", "{noun} v{num}", "Your order #{num}", "(no subject)",
            "[JIRA] ({key}-{num}) {noun} broken", "Re: [{list}] {noun}", "Quick question", "Following up"]

_SPLIT = BODY_SENTENCES.index("Le {noun} est prêt.")
BODY_SENTENCES_EN = BODY_SENTENCES[:_SPLIT]
_OTHER = BODY_SENTENCES[_SPLIT:]
BODY_SENTENCES_BY_LOCALE: dict[str, list[str]] = {
    "fr": _OTHER[0:7], "de": _OTHER[7:15], "es": _OTHER[15:22], "it": _OTHER[22:25], "nl": _OTHER[25:28],
    "pt": _OTHER[28:31], "sv": _OTHER[31:33], "pl": _OTHER[33:35], "ja": _OTHER[35:41], "zh": _OTHER[41:47],
}
