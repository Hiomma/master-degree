import math


def Filter_Array(ds_Filter, nr_Index):
    return list(filter(lambda x: ds_Filter == x[nr_Index], objArrayEstados))


def Print_State(objArrayEstados):

    if(len(objArrayEstados) == 1):
        print('\n')
        print('\nO estado que você está pensando é o: {0}'.format(
            objArrayEstados[0][0]))

        return input('\n\nDeseja continuar identificando outros estados (S/N)?')

    elif(len(objArrayEstados) == 0):
        print('\nNão foi encontrado nenhum estados com essas caracteristicas.')

        return input('\n\nDeseja continuar identificando outros estados (S/N)?')


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
    return input('Qual é o bioma predominante no seu Estado?\n')


def Get_Climate_Question():
    print('---------------------------- Climas ----------------------------\n')
    print('Equatorial')
    print('Semiárido')
    print('Tropical')
    print('Subtropical')
    print('----------------------------------------------------------------\n')
    print('\n')
    return input('Qual é o clima predominante no seu Estado?\n')


def Get_Coast_Question():
    return input('Por favor, responda com S ou N se o seu estado é litorâneo:\n')


def Get_Border_Question():
    return input('Por favor, responda com S ou N se o seu estado é fronteira com algum país:\n')


def Get_PIB_Question(vl_PIB):
    return input('O seu estado tem o PIB maior que {0}?\n'.format(vl_PIB))


def Get_Population_Question(vl_Population):
    return input('O seu estado tem a população maior que {0}?\n'.format(vl_Population))


def Get_COVID_Question(vl_COVID):
    return input('O seu estado teve o número de óbitos por COVID  maior que {0}?\n'.format(vl_COVID))


def Get_Medium_Value(objArrayEstados, nr_Index):
    objArrayEstados.sort(key=lambda x: x[nr_Index])

    return objArrayEstados[math.ceil(len(objArrayEstados)/2) - 1][nr_Index] + 1


b_Continue = True
    
print('Bem-vindo ao Descobridor de Estados, irei fazer algumas perguntas e você me fornece as respostas, assim eu descobrirei de qual estado você está pensando. Vamos lá?')

while(b_Continue):

    objArrayEstados = [
        ['Acre', 'Amazonia', 'Equatorial', 'N', 'S', 15331, 906876, 1816],
        ['Alagoas', 'Caatinga', 'Semiárido', 'S', 'N', 54413, 3365351, 6141],
        ['Amapá', 'Amazonia', 'Equatorial', 'S', 'S', 16795, 877613, 1962],
        ['Amazonas', 'Amazonia', 'Equatorial', 'N', 'S', 100109, 4269995, 13705],
        ['Bahia', 'Mata Atlântica', 'Tropical', 'S', 'N', 286240, 14985284, 26650],
        ['Ceará', 'Caatinga', 'Semiárido', 'S', 'N', 155904, 9240580, 24127],
        ['Distrito Federal', 'Cerrado', 'Tropical',
            'N', 'N', 254817, 3094325, 10222],
        ['Espiríto Santo', 'Mata Atlântica', 'Tropical',
            'S', 'N', 137020, 4108508, 12369],
        ['Goiás', 'Cerrado', 'Tropical', 'N', 'N', 195682, 7206589, 22974],
        ['Maranhão', 'Amazonia', 'Equatorial', 'S', 'N', 98179, 7153262, 10106],
        ['Mato Grosso', 'Pantanal', 'Equatorial',
            'N', 'S', 137443, 3567234, 13655],
        ['Mato Grosso do Sul', 'Pantanal', 'Tropical',
            'N', 'S', 106969, 2839188, 9468],
        ['Minas Gerais', 'Cerrado', 'Tropical', 'N', 'N', 614876, 21411923, 53698],
        ['Pará', 'Amazonia', 'Equatorial', 'S', 'S', 440029, 8777124, 38116],
        ['Paraíba', 'Caatinga', 'Semiárido', 'S', 'N', 64374, 4059905, 9246],
        ['Paraná', 'Mata Atlântica', 'Subtropical',
            'S', 'S', 161350, 11597484, 16544],
        ['Pernambuco', 'Caatinga', 'Semiárido', 'S', 'N', 186352, 9674793, 19552],
        ['Piauí', 'Caatinga', 'Semiárido', 'S', 'N', 50378, 3289290, 6980],
        ['Rio de Janeiro' 'Mata Atlântica', 'Tropical',
            'S', 'N', 758859, 17463349, 63880],
        ['Rio Grande do Norte', 'Caatinga', 'Semiárido',
            'S', 'N', 66970, 3560903, 7302],
        ['Rio Grande do Sul', 'Pampa', 'Subtropical',
            'S', 'S', 457294, 11466630, 34462],
        ['Rondônia', 'Amazonia', 'Equatorial', 'N', 'S', 44914, 1815278, 6506],
        ['Roraima', 'Amazonia', 'Equatorial', 'N', 'S', 13370, 652713, 1968],
        ['Santa Catarina', 'Mata Atlântica', 'Subtropical',
            'S', 'S', 298227, 7338473, 18953],
        ['São Paulo', 'Mata Atlântica', 'Tropical',
            'S', 'N', 2210562, 46649132, 147258],
        ['Sergipe', 'Caatinga', 'Semiárido', 'S', 'N', 42018, 2338474, 6003],
        ['Tocantins', 'Cerrado', 'Tropical', 'N', 'N', 35666, 1607363, 3716],
    ]


    ######################################## Biome question ###############################################

    nm_Estado = Get_Biome_Question()
    objArrayEstados = Filter_Array(nm_Estado, 1)

    sn_Response = Print_State(objArrayEstados)
    if sn_Response == 'N':
        b_Continue = False
        break
    elif sn_Response == 'S':
        continue

    ######################################## Climate question ###############################################

    nm_Clima = Get_Climate_Question()
    objArrayEstados = Filter_Array(nm_Clima, 2)

    sn_Response = Print_State(objArrayEstados)
    if sn_Response == 'N':
        b_Continue = False
        break
    elif sn_Response == 'S':
        continue

    ######################################## PIB question ###############################################

    nr_Medium_Vale = Get_Medium_Value(objArrayEstados, 5)
    sn_PIB = Get_PIB_Question(nr_Medium_Vale)
    objArrayEstados = list(filter(lambda x: x[5] > nr_Medium_Vale,  objArrayEstados) if sn_PIB == 'S' else filter(
        lambda x: x[5] < nr_Medium_Vale,  objArrayEstados))

    sn_Response = Print_State(objArrayEstados)
    if sn_Response == 'N':
        b_Continue = False
        break
    elif sn_Response == 'S':
        continue

    ######################################## Coast question ###############################################

    sn_Litoral = Get_Coast_Question()
    objArrayEstados = Filter_Array(sn_Litoral, 3)

    sn_Response = Print_State(objArrayEstados)
    if sn_Response == 'N':
        b_Continue = False
        break
    elif sn_Response == 'S':
        continue

    ######################################## Border question ###############################################

    sn_Fronteira = Get_Border_Question()
    objArrayEstados = Filter_Array(sn_Fronteira, 4)

    sn_Response = Print_State(objArrayEstados)
    if sn_Response == 'N':
        b_Continue = False
        break
    elif sn_Response == 'S':
        continue

    ######################################## Population question ###############################################

    nr_Medium_Vale = Get_Medium_Value(objArrayEstados, 6)
    sn_Population = Get_Population_Question(nr_Medium_Vale)
    objArrayEstados = list(filter(lambda x: x[6] > nr_Medium_Vale,  objArrayEstados) if sn_Population == 'S' else filter(
        lambda x: x[6] < nr_Medium_Vale,  objArrayEstados))

    sn_Response = Print_State(objArrayEstados)
    if sn_Response == 'N':
        b_Continue = False
        break
    elif sn_Response == 'S':
        continue

    ######################################## COVID question ###############################################

    nr_Medium_Vale = Get_Medium_Value(objArrayEstados, 7)
    sn_COVID = Get_COVID_Question(nr_Medium_Vale)
    objArrayEstados = list(filter(lambda x: x[7] > nr_Medium_Vale,  objArrayEstados) if sn_COVID == 'S' else filter(
        lambda x: x[7] < nr_Medium_Vale,  objArrayEstados))

    sn_Response = Print_State(objArrayEstados)
    if sn_Response == 'N':
        b_Continue = False
        break
    elif sn_Response == 'S':
        continue
