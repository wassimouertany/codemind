package com.demo.service;

import org.springframework.stereotype.Service;

@Service
public class PaymentService {

    private static final int TIMEOUT_MS = 3000;

    public void processPayment(double amount) {
        long elapsed = callGateway(amount);
        if (elapsed > TIMEOUT_MS) {
            throw new PaymentTimeoutException("gateway exceeded " + TIMEOUT_MS + "ms");
        }
    }

    private long callGateway(double amount) {
        return System.currentTimeMillis() % 5000;
    }
}
