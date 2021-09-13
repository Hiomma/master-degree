def Filter_Array(ds_Filter, nr_Index):
    return list(filter(lambda x: ds_Filter == x[nr_Index], objArrayEstados))


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

print(len(objArrayEstados))
