
%Accuracy and number of maps of the last conv layer
data = [94.20,80;92.60,120;94.33,64;93.73,56;93.20,72;92.53,96];

% Draw the plot
scatter(data(:,2),data(:,1));
title('Accuracy network 5');
xlabel('number of maps');
ylabel('Accuracy');