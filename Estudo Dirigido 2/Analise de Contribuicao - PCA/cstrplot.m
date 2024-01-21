function cstrplot(N,tempo_inicial,tempo_final)

data = cstrdataread(N,';','normal');


% Vari�veis
x1=cellstr('FEED CONCENTRATION SENSOR [mol/m�]');
x2=cellstr('FEED FLOWRATE SENSOR [m�/s]');
x3=cellstr('FEED TEMPERATURE SENSOR [�C]');
x4=cellstr('REACTOR LEVEL SENSOR [m]');
x5=cellstr('CONCENTRATION PRODUCT "A" SENSOR [mol/m�]');
x6=cellstr('CONCENTRATION PRODUCT "B" SENSOR [mol/m�]');
x7=cellstr('REACTOR TEMPERATURE SENSOR [�C]');
x8=cellstr('COOLING WATER FLOWRATE SENSOR [m�/s]');
x9=cellstr('PRODUCT FLOWRATE SENSOR [m�/s]');
x10=cellstr('COOLING WATER TEMPERATURE SENSOR [�C]');
x11=cellstr('COOLING WATER PRESSURE SENSOR [Pa]');
x12=cellstr('LEVEL CONTROLLER OUTPUT SIGNAL [%]');
x13=cellstr('COOLING WATER FLOW CONTROLLER OUTPUT SIGNAL [%]');
x14=cellstr('COOLING WATER FLOW SETPOINT [m�/s]');
x15=cellstr('INVENTORY [m�]');
x16=cellstr('MOL BALANCE [m�]');
x17=cellstr('COOLING WATER HEAD LOSS [m]');
x18=cellstr('EFFLUENT HEAD LOSS [m]');

x=[x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18];

if nargin == 1
    % Figura 1 vari�veis de 1 a 9
    figure
    for i = 1:1:9
        subplot(3,3,i)
        plot(data.X(:,i))
        title(x(i))
        xlabel('Time [min]')
        xlim([0 length(data.X(:,i))])
    end
    
    % Figura 2 vari�veis de 18 a 18
    figure
    for i = 10:1:18
        subplot(3,3,(i-9))
        plot(data.X(:,i))
        title(x(i))
        xlabel('Time [min]')
        xlim([0 length(data.X(:,i))])
    end
end

if nargin == 2
    t_i = tempo_inicial;
    
    % Figura 1 vari�veis de 1 a 9
    figure
    for i = 1:1:9
        subplot(3,3,i)
        plot(data.X(:,i))
        title(x(i))
        xlabel('Time [min]')
        xlim([t_i length(data.X(:,i))])
    end
    
    % Figura 2 vari�veis de 18 a 18
    figure
    for i = 10:1:18
        subplot(3,3,(i-9))
        plot(data.X(:,i))
        title(x(i))
        xlabel('Time [min]')
        xlim([t_i-1 length(data.X(:,i))])
    end
end    
 
if nargin == 3
    t_i = tempo_inicial;
    t_f = tempo_final;
    
    % Figura 1 vari�veis de 1 a 9
    figure
    for i = 1:1:9
        subplot(3,3,i)
        plot(data.X(:,i))
        title(x(i))
        xlabel('Time [min]')
        xlim([t_i t_f])
    end
    
    % Figura 2 vari�veis de 18 a 18
    figure
    for i = 10:1:18
        subplot(3,3,(i-9))
        plot(data.X(:,i))
        title(x(i))
        xlabel('Time [min]')
        xlim([t_i t_f])
    end
end
    
end