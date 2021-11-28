import pandas as pd
import numpy as np

dataset = pd.read_csv("datasets/pogo.csv")
NAMES = set(dataset.Name.values)
CPMS = np.asarray([0.09399999678134918, 0.1351374313235283, 0.16639786958694458, 0.1926509141921997, 0.21573247015476227, 0.23657265305519104, 0.2557200491428375, 0.27353037893772125, 0.29024988412857056, 0.3060573786497116, 0.3210875988006592, 0.33544503152370453, 0.3492126762866974, 0.362457737326622, 0.37523558735847473, 0.38759241108516856, 0.39956727623939514, 0.4111935495172506, 0.4225000143051148, 0.4329264134104144, 0.443107545375824, 0.4530599538719858, 0.46279838681221, 0.4723360780626535, 0.4816849529743195, 0.4908558102324605, 0.4998584389686584, 0.5087017565965652, 0.517393946647644, 0.5259425118565559, 0.5343543291091919, 0.5426357612013817, 0.5507926940917969, 0.5588305993005633, 0.5667545199394226, 0.574569147080183, 0.5822789072990417, 0.5898879119195044, 0.5974000096321106, 0.6048236563801765, 0.6121572852134705, 0.6194041110575199, 0.6265671253204346, 0.633649181574583, 0.6406529545783997, 0.6475809663534164, 0.654435634613037, 0.6612192690372467, 0.667934000492096, 0.6745819002389908, 0.6811649203300476, 0.6876849085092545, 0.6941436529159546, 0.7005428969860077, 0.7068842053413391, 0.7131690979003906, 0.719399094581604, 0.7255756109952927, 0.7317000031471252, 0.7347410172224045, 0.7377694845199585, 0.740785576403141, 0.7437894344329834, 0.7467812150716782, 0.7497610449790955, 0.7527291029691696, 0.7556855082511902, 0.7586303651332855, 0.7615638375282288, 0.7644860669970512, 0.7673971652984619, 0.7702972739934921, 0.7731865048408508, 0.7760649472475052, 0.7789327502250671, 0.78179006, 0.78463697, 0.78747358, 0.79030001])

def fix_name(inp):
    # len(names) ~ 500
    for name in NAMES:
        if name in inp:
            return name
    return inp

def ivs_to_cp(name, iv_stamina, iv_attack, iv_defense):
    if name not in NAMES:
        return []
    # reference = https://www.reddit.com/r/TheSilphArena/comments/aa3e1z/guide_primer_on_stats_and_level_when_constrained/
    sel = dataset.Name == name
    pokemon_values = dataset.loc[sel].to_dict(orient='records')[0]

    base_stamina, base_attack, base_defense = pokemon_values['Stamina'], pokemon_values['Attack'], pokemon_values['Defense']


    attack = (iv_attack + base_attack) * CPMS
    defense = (iv_defense + base_defense) * CPMS
    stamina = (iv_stamina + base_stamina) * CPMS

    cp = (np.sqrt(defense) * np.sqrt(stamina) * attack)/10

    return np.floor(cp).astype(int)

def cp_to_t(cp):
    pass

if __name__ == "__main__":
    print(14 in ivs_to_cp("Mudkip", 10, 10, 10))