% Creazione dell'albero decisionale casuale con 5 nodi
albero = struct('testo', [], 'risposta1', [], 'risposta2', []);
albero(1).testo = 'Hai fame?';
albero(1).risposta1 = struct('testo', 'Mangia una pizza', 'risposta1', [], 'risposta2', []);
albero(1).risposta2 = struct('testo', 'Mangia una mela', 'risposta1', [], 'risposta2', []);
albero(2).testo = 'Stai cercando un lavoro?';
albero(2).risposta1 = struct('testo', 'Cerca su LinkedIn', 'risposta1', [], 'risposta2', []);
albero(2).risposta2 = struct('testo', 'Cerca su Glassdoor', 'risposta1', [], 'risposta2', []);
albero(3).testo = 'Hai voglia di fare una passeggiata?';
albero(3).risposta1 = struct('testo', 'Vai al parco', 'risposta1', [], 'risposta2', []);
albero(3).risposta2 = struct('testo', 'Vai in centro', 'risposta1', [], 'risposta2', []);
albero(4).testo = 'Hai bisogno di riposare?';
albero(4).risposta1 = struct('testo', 'Fai una siesta', 'risposta1', [], 'risposta2', []);
albero(4).risposta2 = struct('testo', 'Fai una camminata', 'risposta1', [], 'risposta2', []);
albero(5).testo = 'Vuoi imparare qualcosa di nuovo?';
albero(5).risposta1 = struct('testo', 'Impara a suonare uno strumento', 'risposta1', [], 'risposta2', []);
albero(5).risposta2 = struct('testo', 'Impara una nuova lingua', 'risposta1', [], 'risposta2', []);

% Iterazione con l'albero decisionale
currentNode = 1;
while ~isempty(albero(currentNode).testo)
    disp(albero(currentNode).testo);
    risposta = input('(s/n): ', 's');
    if strcmpi(risposta, 's')
        %currentNode = find(~isempty({albero(currentNode).risposta1.testo}), 1);
        disp(albero(currentNode).risposta1.testo)
        currentNode = currentNode * 2;
        disp(currentNode)
    elseif strcmpi(risposta, 'n')
        currentNode = find(~isempty({albero(currentNode).risposta2.testo}), 1);
        disp(albero(currentNode).risposta1.testo)
        currentNode = currentNode * 2 + 1;
        disp(currentNode)
    end
end
disp(albero(currentNode).testo);
