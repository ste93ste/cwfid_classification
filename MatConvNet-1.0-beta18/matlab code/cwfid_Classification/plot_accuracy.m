% Accuracy of the networks
data = [94,1;94.07,2;92.53,3;94.13,4;94.20,5;92.33,6];

% Draw the plot
scatter(data(:,2),data(:,1));
title('Accuracy of the network');
xlabel('Network');
ylabel('Accuracy');

