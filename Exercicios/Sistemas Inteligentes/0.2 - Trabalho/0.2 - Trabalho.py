import math


def Filter_Array(ds_Filter, nr_Index):
    return list(filter(lambda x: ds_Filter == x[nr_Index], objArrayEstados))


def Print_State(objArrayEstados):
    print('\n')
    print('O estado que você está pensando é o: {0}'.format(
        objArrayEstados[0][0]))


def Get_Biome_Question():
    print('---------------------------- Biomas ----------------------------\n')
    print('Amazonia')
    print('Caatinga')
    print('Mata Atlântica')
    print('Pantanal')
    print('Cerrado')
    print('Pampa')
    print('----------------------------------------------------------------\n')
    print('\n')
    return input('Qual é o bioma predominante no seu Estado?')


def Get_Climate_Question():
    print('---------------------------- Climas ----------------------------\n')
    print('Equatorial')
    print('Semiárido')
    print('Tropical')
    print('Subtropical')
    print('----------------------------------------------------------------\n')
    print('\n')
    return input('Qual é o clima predominante no seu Estado?')


def Get_Coast_Question():
    return input('Por favor, responda com S ou N se o seu estado é litorâneo: ')


def Get_Border_Question():
    return input('Por favor, responda com S ou N se o seu estado é fronteira com algum país: ')


def Get_PIB_Question(vl_PIB):
    return input('O seu estado tem o PIB maior que {0}?'.format(vl_PIB))


def Get_Medium_Value(objArrayEstados):
    objArrayEstados.sort(key=lambda x: x[5])
    return objArrayEstados[math.ceil(len(objArrayEstados)/2)][5] - 1


objArrayEstados = [
    ['Acre', 'Amazonia', 'Equatorial', 'N', 'S', 15331],
    ['Alagoas', 'Caatinga', 'Semiárido', 'S', 'N', 54413],
    ['Amapá', 'Amazonia', 'Equatorial', 'S', 'S', 16795],
    ['Amazonas', 'Amazonia', 'Equatorial', 'N', 'S', 100109],
    ['Bahia', 'Mata Atlântica', 'Tropical', 'S', 'N', 286240],
    ['Ceará', 'Caatinga', 'Semiárido', 'S', 'N', 155904],
    ['Distrito Federal', 'Cerrado', 'Tropical', 'N', 'N', 254817],
    ['Espiríto Santo', 'Mata Atlântica', 'Tropical', 'S', 'N', 137020],
    ['Goiás', 'Cerrado', 'Tropical', 'N', 'N', 195682],
    ['Maranhão', 'Amazonia', 'Equatorial', 'S', 'N', 98179],
    ['Mato Grosso', 'Pantanal', 'Equatorial', 'N', 'S', 137443],
    ['Mato Grosso do Sul', 'Pantanal', 'Tropical', 'N', 'S', 106969],
    ['Minas Gerais', 'Cerrado', 'Tropical', 'N', 'N', 614876],
    ['Pará', 'Amazonia', 'Equatorial', 'S', 'S', 440029],
    ['Paraíba', 'Caatinga', 'Semiárido', 'S', 'N', 64374],
    ['Paraná', 'Mata Atlântica', 'Subtropical', 'S', 'S', 161350],
    ['Pernambuco', 'Caatinga', 'Semiárido', 'S', 'N', 186352],
    ['Piauí', 'Caatinga', 'Semiárido', 'S', 'N', 50378],
    ['Rio de Janeiro' 'Mata Atlântica', 'Tropical', 'S', 'N', 758859],
    ['Rio Grande do Norte', 'Caatinga', 'Semiárido', 'S', 'N', 66970],
    ['Rio Grande do Sul', 'Pampa', 'Subtropical', 'S', 'S', 457294],
    ['Rondônia', 'Amazonia', 'Equatorial', 'N', 'S', 44914],
    ['Roraima', 'Amazonia', 'Equatorial', 'N', 'S', 13370],
    ['Santa Catarina', 'Mata Atlântica', 'Subtropical', 'S', 'S', 298227],
    ['São Paulo', 'Mata Atlântica', 'Tropical', 'S', 'N', 2210562],
    ['Sergipe', 'Caatinga', 'Semiárido', 'S', 'N', 42018],
    ['Tocantins', 'Cerrado', 'Tropical', 'N', 'N', 35666],
]

print('Bem-vindo ao Descobridor de Estados, irei fazer algumas perguntas e você me fornece as respostas, assim eu descobrirei de qual estado você está pensando. Vamos lá?')

nm_Estado = Get_Biome_Question()
objArrayEstados = Filter_Array(nm_Estado, 1)

nm_Clima = Get_Climate_Question()
objArrayEstados = Filter_Array(nm_Clima, 2)

nr_Medium_Vale = Get_Medium_Value(objArrayEstados)
sn_PIB = Get_PIB_Question(nr_Medium_Vale)
objArrayEstados = list(filter(lambda x: x[5] > nr_Medium_Vale,  objArrayEstados) if sn_PIB == 'S' else filter(
    lambda x: x[5] < nr_Medium_Vale,  objArrayEstados))

sn_Litoral = Get_Coast_Question()
objArrayEstados = Filter_Array(sn_Litoral, 3)

sn_Fronteira = Get_Border_Question()
objArrayEstados = Filter_Array(sn_Fronteira, 4)

Print_State(objArrayEstados)
