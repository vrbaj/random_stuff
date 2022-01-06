for A in range(8):
    for B in range(8):
        for sn in range(2):
            s = bool(sn) #značí odčítání
            Ab = bin(A)[2:].zfill(4) #Doplňkový kód zobrazuje kladná čísla na sebe sama, takže je stačí převést na binární pomocí přímého kódu
            Bb = bin(B)[2:].zfill(4)
            C = s # Carry bit
            Sb = "" #Suma v binární reprezentaci
            for ii in range(4):   #Každá iterace znázorňuje jeden full adder včetně implementace odčítání
                i = 3-ii #vzhledem k reprezentaci musíme postupovat ve stringu odzadu
                Ain = bool(Ab[i])
                Cin = C
                Bin = (bool(Bb[i]) != s)
                Sout = ((Ain != Bin) != Cin)
                C = (Ain and Bin) or (Cin and Bin) or (Cin and Ain)
                Sb = str(int(Sout)) +Sb


            # Kontrola funkce sčítačky
            Scontrol = A + (-1 if s else 1)*B
            S = int(Sb,2) if Sout else int(Sb,2) - (1 << 4) # Převod výsledku z doplňkového kódu na int
            if S != Scontrol:
                print("Sčítačka nefunguje")
                print("A: " + str(A))
                print("B: " + str(B))
                print("Odčítám" if s else "Sčítám")
                print("Suma: " + str(S))
                print("Kontrola: "+ str(Scontrol))