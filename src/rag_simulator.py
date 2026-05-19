"""
rag_simulator.py — Gerador de contexto RAG massivo com terminologia médica.

Produz um documento fictício de 10.000-15.000 tokens simulando um compêndio
médico multi-capítulo, adequado para estressar o pipeline de atenção do LLM.
"""

import random
from typing import Optional

# ---------------------------------------------------------------------------
# Blocos de texto médico estruturado
# ---------------------------------------------------------------------------

_CHAPTERS = [
    {
        "title": "Capítulo 1 — Fisiopatologia das Doenças Cardiovasculares",
        "sections": [
            (
                "1.1 Aterosclerose e Doença Arterial Coronariana",
                """A aterosclerose é uma doença inflamatória crônica das artérias caracterizada pelo
acúmulo de lipídeos, células inflamatórias e tecido fibroso na parede arterial, formando
placas ateroscleróticas que comprometem progressivamente o lúmen vascular. O processo
inicia-se com a disfunção endotelial, precipitada por fatores como hipertensão arterial
sistêmica, dislipidemia, tabagismo, diabetes mellitus e estresse oxidativo. As lipoproteínas
de baixa densidade (LDL) oxidadas penetram o subendotélio e são fagocitadas por macrófagos,
originando as células espumosas (foam cells). A resposta inflamatória local recruta monócitos
e linfócitos T, que secretam citocinas pró-inflamatórias como TNF-α, IL-1β e IL-6,
amplificando o processo. A estabilidade da placa depende da espessura da cápsula fibrosa;
placas vulneráveis com cápsula delgada e núcleo lipídico extenso estão sujeitas à ruptura,
precipitando trombose aguda e síndromes coronarianas agudas — angina instável, infarto agudo
do miocárdio com e sem supradesnivelamento do segmento ST (IAMCSSST e IAMSSST).

O diagnóstico é suportado por eletrocardiografia (ECG), biomarcadores séricos (troponina I
e T de alta sensibilidade, CK-MB), ecocardiografia bidimensional, cineangiocoronariografia e,
crescentemente, por tomografia computadorizada de artérias coronárias com escore de cálcio
(CACS). Scores como GRACE e TIMI estratificam o risco e orientam a terapia de
revascularização percutânea (ICP) ou cirúrgica (CABG).""",
            ),
            (
                "1.2 Insuficiência Cardíaca — Mecanismos Moleculares",
                """A insuficiência cardíaca (IC) representa a via final comum de inúmeras cardiopatias,
caracterizando-se pela incapacidade do coração de suprir a demanda metabólica tissular a
pressões de enchimento normais. Classifica-se em IC com fração de ejeção reduzida (ICFEr,
FEVE < 40%), IC com fração de ejeção intermediária (ICFEi, FEVE 40–49%) e IC com fração de
ejeção preservada (ICFEp, FEVE ≥ 50%). A fisiopatologia envolve ativação neurohormonal do
sistema renina-angiotensina-aldosterona (SRAA) e do sistema nervoso simpático, com aumento
das catecolaminas circulantes, retenção hidrossalina e remodelamento cardíaco adverso.

Em nível molecular, a sobrecarga hemodinâmica ativa vias de sinalização hipertrófica
(calcineurina/NFAT, MAPK/ERK) e apoptótica (mitocondrial via citocromo c e caspases). O
remodelamento da matriz extracelular é mediado por metaloproteinases (MMPs) e seus inibidores
teciduais (TIMPs), culminando em fibrose intersticial e disfunção diastólica. Biomarcadores
como peptídeos natriuréticos (BNP, NT-proBNP) refletem a tensão parietal e orientam o
diagnóstico e monitoramento terapêutico. O tratamento da ICFEr baseia-se em inibidores da
SRAA (IECA ou BRA), betabloqueadores, antagonistas da aldosterona (espironolactona, eplerenona),
inibidores SGLT2 (dapagliflozina, empagliflozina) e sacubitril/valsartana.""",
            ),
        ],
    },
    {
        "title": "Capítulo 2 — Neurologia: Doenças Neurodegenerativas",
        "sections": [
            (
                "2.1 Doença de Alzheimer — Hipóteses Patogênicas Atuais",
                """A doença de Alzheimer (DA) é a causa mais prevalente de demência, respondendo por
60–70% dos casos globais. Histopatologicamente, define-se pela presença de placas senis
(depósitos extracelulares de peptídeo β-amiloide, Aβ42) e emaranhados neurofibrilares
(agregados intracelulares de proteína tau hiperfosforilada). A hipótese amiloide postula que
o desequilíbrio entre produção e clearance de Aβ é o evento iniciador da cascata
neurodegenerativa. A clivagem amiloidogênica da proteína precursora amiloide (APP) pela
β-secretase (BACE1) e pelo complexo γ-secretase (presenilina 1/2) gera Aβ40 e Aβ42; o
Aβ42, mais hidrofóbico, tem maior propensão à oligomerização e deposição.

Oligômeros solúveis de Aβ são atualmente considerados as espécies mais neurotóxicas,
comprometendo a sinalização sináptica, ativando resposta inflamatória glial (microgliose e
astrogliose) e desencadeando morte neuronal via estresse oxidativo e disfunção mitocondrial.
A propagação da patologia tau segue os critérios de estadiamento de Braak (I–VI). Biomarcadores
no líquido cefalorraquidiano (LCR) — Aβ42 reduzido, tau total e tau fosforilada aumentadas —
e PET amiloide/tau são ferramentas diagnósticas centrais. Terapias aprovadas incluem
anticorpos anti-amiloide (lecanemab, donanemab), inibidores de acetilcolinesterase
(donepezila, rivastigmina, galantamina) e memantina (antagonista NMDA).""",
            ),
            (
                "2.2 Doença de Parkinson — Disfunção Dopaminérgica e α-Sinucleína",
                """A doença de Parkinson (DP) é uma sinucleinopatia progressiva caracterizada pela
degeneração de neurônios dopaminérgicos na substância negra pars compacta (SNpc) e pela
presença de corpos de Lewy (inclusões intracelulares ricas em α-sinucleína). A perda superior
a 70–80% dos neurônios dopaminérgicos precipita o aparecimento dos sinais motores cardinais:
tremor de repouso (4–6 Hz), rigidez em roda denteada, bradicinesia e instabilidade postural.
Sintomas não-motores frequentemente precedem a fase motora: hiposmia, constipação, distúrbio
comportamental do sono REM e depressão.

A α-sinucleína, proteína pré-sináptica normalmente solúvel, sofre misfolding e forma fibrilas
amiloides que se propagam de neurônio em neurônio por mecanismo prion-like, explicando o
estadiamento de Braak (estágios 1–6). Mutações nos genes SNCA, LRRK2, GBA, PINK1, PARKIN e
DJ-1 identificam formas genéticas e iluminam vias patogênicas (disfunção lisossomal,
mitofagia, estresse do retículo endoplasmático). O tratamento dopaminérgico — levodopa/carbidopa,
agonistas dopaminérgicos (pramipexol, ropinirol, rotigotina), inibidores de MAO-B
(rasagilina, safinamida) e inibidores de COMT (entacapona) — controla sintomas motores sem
modificar a progressão da doença.""",
            ),
            (
                "2.3 Esclerose Lateral Amiotrófica — Mecanismos e Perspectivas Terapêuticas",
                """A esclerose lateral amiotrófica (ELA) é uma doença fatal do neurônio motor caracterizada
pela degeneração simultânea do neurônio motor superior (córtex motor → trato corticoespinal)
e inferior (corno anterior medular e núcleos motores do tronco encefálico). A incidência é de
2–3 casos/100.000/ano; a sobrevida mediana após o diagnóstico é de 2–5 anos, sendo a
insuficiência respiratória a principal causa de óbito. A maioria (90%) dos casos é esporádica;
mutações em SOD1, C9orf72 (expansão hexanucleotídica GGGGCC), TDP-43 e FUS explicam
parcelas significativas dos casos familiares.

Os mecanismos patogênicos incluem: excitotoxicidade glutamatérgica (justificando o riluzol —
único modificador de doença de uso oral aprovado antes de 2023), agregação proteica, disfunção
mitocondrial, defeitos do transporte axonal e neuroinflamação. Edaravona (eliminador de
radicais livres) e tofersen (oligonucleotídeo antissentido para SOD1) compõem o arsenal
terapêutico atual. Pesquisas em terapia gênica, células-tronco mesenquimais e estratégias
de edição CRISPR representam fronteiras em investigação clínica.""",
            ),
        ],
    },
    {
        "title": "Capítulo 3 — Oncologia Molecular: Biologia do Câncer",
        "sections": [
            (
                "3.1 Oncogênese e Hallmarks do Câncer",
                """O câncer resulta da acumulação de alterações somáticas que conferem às células
vantagens proliferativas e de sobrevivência. Os hallmarks clássicos de Hanahan e Weinberg
(2000, revisados em 2011 e 2022) incluem: autossuficiência em sinais de crescimento, evasão
de supressores de crescimento, resistência à apoptose, potencial replicativo ilimitado
(ativação da telomerase), indução de angiogênese e ativação de invasão/metástase. Hallmarks
emergentes incluem: reprogramação do metabolismo energético (efeito Warburg — preferência
por glicólise anaeróbica mesmo em normóxia), evasão da imunovigilância, inflamação promotora
de tumor, instabilidade genômica e mutação, plasticidade fenotípica e microbioma tumoral.

Oncogenes (RAS, MYC, EGFR, HER2, BCR-ABL, PIK3CA) e genes supressores de tumor (TP53, RB1,
BRCA1/2, PTEN, APC, VHL) constituem os alvos moleculares centrais. O sequenciamento de
nova geração (NGS) — painéis de genes, sequenciamento de exoma inteiro (WES) e genoma inteiro
(WGS) — permite caracterização mutacional abrangente. A carga mutacional tumoral (TMB) e a
instabilidade de microssatélites (MSI-H/dMMR) são biomarcadores preditivos de resposta à
imunoterapia com inibidores de checkpoint (anti-PD-1, anti-PD-L1, anti-CTLA-4).""",
            ),
            (
                "3.2 Imunoterapia Oncológica e Resistência",
                """A imunoterapia revolucionou o tratamento de múltiplas neoplasias malignas. Os inibidores
de checkpoint imunológico (ICIs) — pembrolizumab, nivolumab, atezolizumab, durvalumab
(anti-PD-1/PD-L1) e ipilimumab (anti-CTLA-4) — restauram a capacidade citotóxica dos
linfócitos T tumorais ao bloquear mecanismos de evasão imune. As terapias CAR-T (Chimeric
Antigen Receptor T-cell) rediagram linfócitos T autólogos para reconhecer antígenos tumorais
específicos (CD19 em linfomas B, BCMA em mieloma múltiplo), alcançando remissões completas
em doenças previamente refratárias.

A resistência à imunoterapia pode ser primária (ausência de resposta inicial) ou adquirida
(progressão após resposta inicial). Mecanismos incluem: perda de expressão de neo-antígenos
tumorais (por deleção genômica ou downregulation de MHC-I), upregulation de checkpoints
alternativos (TIM-3, LAG-3, TIGIT), imunossupressão mediada pelo microambiente tumoral (TME)
via células Treg, macrófagos M2, MDSCs e citocinas imunomoduladoras (IL-10, TGF-β, VEGF),
e alterações em vias de sinalização intrínsecas ao tumor (perda de PTEN, ativação de
β-catenina/WNT). Combinações de ICIs, vacinas tumorais e terapias direcionadas são exploradas
para superar a resistência.""",
            ),
        ],
    },
    {
        "title": "Capítulo 4 — Infectologia: Patógenos Emergentes e Resistência Antimicrobiana",
        "sections": [
            (
                "4.1 Resistência Antimicrobiana — Mecanismos e Epidemiologia",
                """A resistência antimicrobiana (RAM) constitui uma das mais graves ameaças à saúde global,
estimando-se 1,27 milhão de mortes atribuíveis diretamente em 2019 (Lancet, 2022). Os
mecanismos bacterianos de resistência são classificados em: (1) inativação enzimática —
β-lactamases (incluindo ESBLs, KPC, NDM, OXA-48), aminoglicosídeo-modificadoras,
cloranfenicol-acetiltransferases; (2) modificação do alvo — PBPs alteradas em MRSA
(mecA/mecC), mutações em RNA polimerase (rifampicina), DNA-girase/topoisomerase IV
(fluoroquinolonas), rRNA 23S (macrólidos); (3) efluxo ativo — sistemas RND (MexAB-OprM em
Pseudomonas, AcrAB-TolC em Enterobacterales), SMR, MFS, ABC; (4) redução da permeabilidade
— perda de porinas (OmpC, OmpF); (5) proteção do alvo — proteínas Qnr, metilases rRNA.

A disseminação horizontal de genes de resistência via plasmídeos, transposons e integrons
acelera a crise. Patógenos ESKAPE (Enterococcus faecium, Staphylococcus aureus, Klebsiella
pneumoniae, Acinetobacter baumannii, Pseudomonas aeruginosa, Enterobacter spp.) concentram
o maior impacto clínico. Estratégias de controle englobam: stewardship antimicrobiano,
diagnóstico rápido por PCR multiplex e MALDI-TOF MS, combinações de antibióticos (polimixina
+ meropeném + fosfomicina em Enterobacterales produtoras de carbapenemase), e novos agentes
como cefiderocol, ceftazidima-avibactam, imipeném-cilastatina-relebactam.""",
            ),
            (
                "4.2 Virologia de Vírus RNA — Mecanismos de Patogenia e Resposta Imune Antiviral",
                """Vírus de RNA de sentido positivo, negativo e dupla-fita constituem uma porção substancial
dos patógenos emergentes humanos. Coronavírus (SARS-CoV-2), Flavivírus (Dengue, Zika, febre
amarela), Togavírus (Chikungunya), Filovírus (Ebola, Marburg) e Orthomyxovírus (Influenza)
exemplificam a diversidade das ameaças. O ciclo replicativo de vírus RNA depende de RNA
polimerase dependente de RNA (RdRp) viral, ausente em células humanas e alvo terapêutico
privilegiado (remdesivir, molnupiravir, favipiravir).

A resposta imune inata antiviral central envolve reconhecimento de padrões moleculares
associados a patógenos (PAMPs) — RNA de fita dupla, CpG — por receptores de reconhecimento
de padrões (PRRs): TLR3, TLR7/8 (endossomais) e RIG-I/MDA5 (citoplasmáticos). A ativação
dessas vias converge em IRF3/IRF7 e NF-κB, culminando na produção de interferons tipo I
(IFN-α/β) e citocinas pró-inflamatórias. IFN tipo I sinaliza via STAT1/STAT2/IRF9 (complexo
ISGF3), induzindo centenas de genes estimulados por interferon (ISGs) que estabelecem estado
antiviral celular. Vírus altamente patogênicos desenvolveram estratégias de evasão: NS5A e NS5B
do HCV, ORF6 do SARS-CoV-2, VP35 do Ebola bloqueiam essa cascata em pontos estratégicos.""",
            ),
        ],
    },
    {
        "title": "Capítulo 5 — Endocrinologia: Diabetes Mellitus e Síndrome Metabólica",
        "sections": [
            (
                "5.1 Fisiopatologia do Diabetes Mellitus Tipo 2",
                """O diabetes mellitus tipo 2 (DM2) resulta da combinação de resistência à ação da insulina
em tecidos periféricos (fígado, músculo esquelético, tecido adiposo) e falência progressiva das
células β pancreáticas. A resistência insulínica é favorecida por obesidade visceral, estilo
de vida sedentário, genética e inflamação crônica de baixo grau. O tecido adiposo visceral
secreta adipocinas pró-inflamatórias (TNF-α, IL-6, resistina, visfatina) e reduz a produção
de adiponectina — adipocina insulinossensibilizante — comprometendo a sinalização do receptor
de insulina (IRS-1/2 → PI3K → AKT → GLUT4).

A célula β compensa inicialmente com hipersecreção de insulina; a falência sobrevém por
glicotoxicidade, lipotoxicidade, estresse oxidativo e do retículo endoplasmático, inflamação
mediada por IL-1β/NLRP3 e apoptose. O controle glicêmico baseia-se no octeto DeFronzo:
resistência hepática (aumento da gliconeogênese), resistência muscular (redução da captação
de glicose), disfunção de células β e α (hipersecreção de glucagon), aumento da reabsorção
renal de glicose (cotransportadores SGLT2), incretinas defeituosas (redução de GLP-1),
apetite desregulado e neurotransmissão alterada. Classes terapêuticas modernas atuam nesses
octetos: metformina (fígado), SGLT2i (rim), GLP-1 RAs (pâncreas/cérebro/trato GI), TZDs
(músculo/adiposo), DPP-4i (incretinas), sulfoniluréias/glinidas (células β).""",
            ),
            (
                "5.2 Síndrome Metabólica e Risco Cardiovascular",
                """A síndrome metabólica (SM) é um aglomerado de fatores de risco — obesidade abdominal
(CA ≥ 90 cm homens / ≥ 80 cm mulheres, critério IDF para latinos), hipertrigliceridemia
(≥ 150 mg/dL), HDL baixo (< 40 mg/dL H / < 50 mg/dL M), hipertensão arterial
(≥ 130/85 mmHg) e glicemia de jejum elevada (≥ 100 mg/dL) — que confere risco 2–3 vezes
maior de DM2 e 1,5–2 vezes maior de eventos cardiovasculares maiores (MACE) em relação a
indivíduos sem a síndrome. O substrato fisiopatológico é a resistência à insulina associada
à adiposidade visceral disfuncional.

O fígado gorduroso não alcoólico (NAFLD/MASLD) representa a manifestação hepática da SM,
espectro que vai de esteatose simples à esteato-hepatite (MASH), cirrose e hepatocarcinoma.
Biomarcadores emergentes incluem FIB-4, escore de fibrose hepática, elastografia hepática
(FibroScan) e proteínas do LCR como M-CSF. Intervenção intensiva no estilo de vida reduz
em 58% a progressão para DM2 (DPP, NEJM 2002). Farmacoterapia da MASH inclui semaglutida
2,4 mg/semana (resolução da esteato-hepatite em ~59%, NEJM 2021) e resmetirom (agonista
seletivo de receptor de hormônio tireoidiano β, aprovado FDA 2024 para MASH com fibrose F2-F3).""",
            ),
        ],
    },
    {
        "title": "Capítulo 6 — Pneumologia: Doenças Respiratórias Crônicas",
        "sections": [
            (
                "6.1 DPOC — Doença Pulmonar Obstrutiva Crônica",
                """A doença pulmonar obstrutiva crônica (DPOC) é uma condição prevenível e tratável
caracterizada por limitação persistente e progressiva ao fluxo aéreo, associada a resposta
inflamatória crônica das vias aéreas e pulmões a partículas e gases nocivos — primariamente
fumaça de tabaco (80–90% dos casos), mas também exposição a biomassa, poeira ocupacional e
poluição atmosférica. O diagnóstico espirométrico baseia-se na relação VEF1/CVF pós-broncodilatador
< 0,70 (critério GOLD). O estadiamento GOLD (1–4) classifica a gravidade pela redução do VEF1:
GOLD 1 (≥ 80%), GOLD 2 (50–79%), GOLD 3 (30–49%), GOLD 4 (< 30% predito).

A fisiopatologia envolve: inflamação de vias aéreas por neutrófilos, macrófagos e linfócitos
CD8+; hiperinsuflação dinâmica por aprisionamento aéreo (aumento do VR e CRF); destruição do
parênquima (enfisema) por desequilíbrio protease/antiprotease e estresse oxidativo; remodelamento
de vias aéreas por hipertrofia de glândulas mucosas e metaplasia escamosa. A prevenção de
exacerbações é o principal objetivo terapêutico; tripla terapia inalatória (LABA + LAMA + ICS)
reduz exacerbações em 25% em relação a LABA+LAMA. Eosinófilos sanguíneos (≥ 300/μL) predizem
resposta a ICS. Azitromicina profilática e roflumilaste (inibidor PDE4) reduzem exacerbações
em fenótipos específicos.""",
            ),
            (
                "6.2 Asma Brônquica — Mecanismos Imunológicos e Terapia Biológica",
                """A asma brônquica é uma doença inflamatória crônica das vias aéreas marcada por
hiperresponsividade brônquica, obstrução variável ao fluxo aéreo e sintomas recorrentes de
dispneia, sibilância, opressão torácica e tosse. A inflamação eosinofílica Th2-mediada é o
fenótipo mais prevalente: células dendríticas apresentam alérgenos a linfócitos Th2, que
secretam IL-4, IL-5 e IL-13 — promovendo produção de IgE (IL-4), eosinofilopoiese e
recrutamento (IL-5), metaplasia gobelet e hiper-secreção de muco (IL-13). IgE liga-se a
receptores FcεRI em mastócitos e basófilos; a reexposição ao alérgeno precipita degranulação
com liberação de histamina, leucotrienos (LTC4, LTD4) e prostaglandinas.

A classificação GINA (2024) estratifica o controle e o tratamento em escada terapêutica.
Biológicos direcionados a alvos da cascata Th2 transformaram o manejo da asma grave:
omalizumab (anti-IgE), mepolizumab e reslizumab (anti-IL-5), benralizumab (anti-IL-5Rα),
dupilumab (anti-IL-4Rα, bloqueia IL-4 e IL-13), tezepelumab (anti-TSLP — age no topo da
cascata inflamatória, eficaz em fenótipos eosinofílico e não-eosinofílico). A monitorização
de biomarcadores — IgE total, eosinófilos no sangue e escarro, FeNO — orienta a seleção e
reavaliação do biológico adequado.""",
            ),
        ],
    },
    {
        "title": "Capítulo 7 — Hematologia: Doenças do Sangue e Hemostasia",
        "sections": [
            (
                "7.1 Leucemias Agudas — Classificação Molecular e Tratamento",
                """As leucemias agudas são neoplasias hematopoiéticas caracterizadas pela proliferação
clonal e bloqueio de diferenciação de precursores hematopoiéticos. A leucemia mieloide aguda
(LMA) apresenta incidência de 3–4 casos/100.000/ano, com sobrevida global em 5 anos de 30–40%
em adultos. A classificação WHO 2022 integra dados morfológicos, citogenéticos e mutacionais.
Mutações em NPM1 (25–30% dos casos), FLT3-ITD (25–30%), IDH1/IDH2 (15–20%), CEBPA
(10–15%) e RUNX1 definem subgrupos com prognóstico e abordagem terapêuticos distintos.

A indução padrão (3+7: antraciclina por 3 dias + citarabina contínua por 7 dias) é
complementada por agentes alvo-dirigidos: midostaurina ou gilteritinib (FLT3+), ivosidenib/
enasidenib (IDH1/2+), venetoclax (inibidor BCL-2, combinado com azacitidina em pacientes
não candidatos à quimioterapia intensiva — sobrevida global superior ao padrão em idosos).
Gemtuzumab ozogamicin (anti-CD33 conjugado a caliqueamicina) é opção em LMA CD33+. O
transplante alogênico de células-tronco hematopoiéticas (TCTH-alo) permanece como única
terapia curativa consolidada em doença de risco intermediário/alto. A leucemia linfoblástica
aguda (LLA) Ph+ (BCR-ABL1+) é tratada com quimioterapia combinada a inibidores de tirosina-
quinase (imatinibe, dasatinibe, ponatinibe); blinatumomab (BiTE anti-CD19xCD3) e inotuzumab
ozogamicin revolucionaram o tratamento da LLA B refratária/recaída.""",
            ),
        ],
    },
    {
        "title": "Capítulo 8 — Nefrologia: Injúria Renal Aguda e Doença Renal Crônica",
        "sections": [
            (
                "8.1 Injúria Renal Aguda — Critérios, Biomarcadores e Prevenção",
                """A injúria renal aguda (IRA) define-se pelos critérios KDIGO (2012): aumento de creatinina
sérica ≥ 0,3 mg/dL em 48h, elevação ≥ 1,5× o basal em 7 dias ou débito urinário < 0,5 mL/kg/h
por ≥ 6h. Classifica-se em pré-renal (hipoperfusão), intrínseca (tubular — necrose tubular
aguda [NTA], isquêmica ou nefrotóxica; glomerular; intersticial; vascular) e pós-renal
(obstrutiva). A NTA isquêmica, causa mais frequente de IRA em UTIs, resulta da depleção de
ATP com falência das bombas iônicas, disfunção mitocondrial, necrose e apoptose de células
tubulares proximais, inflamação e vasoconstrição renal.

Biomarcadores precoces superam a creatinina em sensibilidade e especificidade: NGAL urinário/
plasmático, KIM-1, IL-18, TIMP-2×IGFBP7 (NephroCheck®, produto > 0,3 prediz IRA KDIGO 2-3
em 12h com AUC ~0,80). O manejo inclui otimização hemodinâmica, ajuste de agentes nefrotóxicos
(aminoglicosídeos, contraste iodado, AINEs, anfotericina B), e suporte renal extracorpóreo
(terapia de substituição renal contínua [CRRT] em hemodinamicamente instáveis ou hemofiltração/
hemodiafiltração com dose efluente ≥ 20 mL/kg/h). A prevenção de IRA por contraste (CIN)
envolve hidratação com salina isotônica e acetilcisteína oral — embora evidências para esta
última sejam controversas.""",
            ),
            (
                "8.2 Doença Renal Crônica — Progressão e Nefroproteção",
                """A doença renal crônica (DRC) define-se pela presença de alterações estruturais ou
funcionais renais persistentes por > 3 meses: TFG < 60 mL/min/1,73m² e/ou albuminúria ≥ 30 mg/g
creatinina. A classificação CKD-EPI 2021 (sem raça) fornece estimativas mais precisas da TFG.
A progressão é determinada pela causa subjacente (nefropatia diabética > hipertensiva >
glomerulonefrites primárias), magnitude da proteinúria (principal fator modificável) e
pressão arterial. A fibrose túbulo-intersticial, via ativação de fibroblastos renais por
TGF-β1 e transição epitélio-mesenquimal, é o substrato histológico final comum.

Estratégias nefroprotetoras de alto nível de evidência: inibidores SGLT2 (dapagliflozina,
empagliflozina) reduzem a progressão da DRC e mortalidade cardiovascular em diabéticos e
não-diabéticos com proteinúria ≥ 200 mg/g (DAPA-CKD, EMPA-KIDNEY); finerenona (ARM
não-esteroidal) reduz desfechos renais e cardiovasculares em DM2+DRC com proteinúria;
IECA/BRA reduzem proteinúria e retardam progressão. O manejo de complicações (anemia —
EPO/HIF-PH inibidores como daprodustat; DRC-mineral-óssea — quelantes de fósforo, vitamina D
ativa, cinacalcete; acidose metabólica — bicarbonato de sódio) e o preparo para terapia renal
substitutiva (hemodiálise, diálise peritoneal ou transplante) são etapas fundamentais do
cuidado nefrológico longitudinal.""",
            ),
        ],
    },
]


def _expand_section(title: str, body: str, repetitions: int = 2) -> str:
    """Expande uma seção para aumentar volume de tokens."""
    content = f"\n{'=' * 70}\n{title}\n{'=' * 70}\n\n"
    for _ in range(repetitions):
        content += body.strip() + "\n\n"
    return content


def generate_medical_corpus(target_chars: int = 55_000) -> str:
    """
    Gera corpus médico multi-capítulo com aproximadamente `target_chars`
    caracteres (~10.000-15.000 tokens para tokenizadores sub-word típicos).
    """
    random.seed(42)
    parts = []
    header = (
        "COMPÊNDIO DE MEDICINA INTERNA — VERSÃO SIMULADA PARA BENCHMARK RAG\n"
        "Documento gerado para fins educacionais e de benchmark de sistemas de IA.\n"
        "Todos os dados clínicos são fictícios. Não utilizar para decisões médicas reais.\n"
        + "=" * 70 + "\n\n"
    )
    parts.append(header)

    for chapter in _CHAPTERS:
        parts.append(f"\n{'#' * 70}\n{chapter['title']}\n{'#' * 70}\n")
        for title, body in chapter["sections"]:
            reps = random.randint(2, 3)
            parts.append(_expand_section(title, body, reps))

    corpus = "".join(parts)

    while len(corpus) < target_chars:
        extra_chapter = random.choice(_CHAPTERS)
        extra_title, extra_body = random.choice(extra_chapter["sections"])
        corpus += _expand_section(
            f"[Revisão Adicional] {extra_title}", extra_body, repetitions=2
        )

    return corpus


def build_rag_prompt(question: str, context: str, max_context_chars: int = 12_000) -> str:
    """Constrói o prompt RAG com contexto truncado para caber no modelo."""
    truncated = context[:max_context_chars]
    return (
        f"Você é um assistente médico especializado. Use o contexto abaixo para "
        f"responder à pergunta de forma precisa e fundamentada.\n\n"
        f"CONTEXTO MÉDICO:\n{truncated}\n\n"
        f"PERGUNTA: {question}\n\n"
        f"RESPOSTA:"
    )
