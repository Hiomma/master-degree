function stat = threshold(M,type)
% function stat = threshold(M,n,type)
% Threshold for T2, Q and combined
% Input:
% M = model
% type of statistic

switch type
    case 't2'
        stat=(M.a*(M.n-1)*(M.n+1)/(M.n*(M.n-M.a)))*finv(M.alfa,M.a,M.n-M.a); 
    case 'q'
        D=diag(M.S);
        m=length(M.mu);
        d=D((M.a+1):m);
        if length(d)==0 stat=[]; return; end; 
        teta1=sum(d);
        teta2=sum(d.^2);
        teta3=sum(d.^3);
        h0=1-(2*teta1*teta3/(3*teta2^2));
        ca=norminv(M.alfa,0,1);
        Q=(ca*sqrt(2*teta2*h0^2)/teta1+1+teta2*h0*(h0-1)/teta1^2);
        stat=teta1*Q^(1/h0);
    case 'c'
        D=diag(M.S);
        m=length(M.mu);
        d=D((M.a+1):m);
        if length(d)==0 stat=[]; return; end; 
        teta1=sum(d);
        teta2=sum(d.^2);
        teta3=sum(d.^3);
        h0=1-(2*teta1*teta3/(3*teta2^2));
        ca=norminv(M.alfa,0,1);
        Q=(ca*sqrt(2*teta2*h0^2)/teta1+1+teta2*h0*(h0-1)/teta1^2);
        Q=teta1*Q^(1/h0);
        t2=(M.a*(M.n-1)*(M.n+1)/(M.n*(M.n-M.a)))*finv(M.alfa,M.a,M.n-M.a);  
        g1=(t2^(-2)+teta2*Q^(-2))/(t2^(-1)+teta1*Q^(-1));
        h1=(t2^(-1)+teta1*Q^(-1))^2/(t2^(-2)+teta2*Q^(-2));
        stat=g1*chi2inv(M.alfa,h1);    

end

end

