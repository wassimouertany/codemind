package com.demo.service;

import com.demo.repository.OrderRepository;
import com.demo.model.Order;
import org.springframework.stereotype.Service;

@Service
public class OrderService {

    private final OrderRepository orderRepository;
    private final PaymentService paymentService;

    public OrderService(OrderRepository orderRepository, PaymentService paymentService) {
        this.orderRepository = orderRepository;
        this.paymentService = paymentService;
    }

    public Order placeOrder(Order order) {
        // BUG: PaymentTimeoutException is unchecked and never handled here,
        // so it propagates to the controller and surfaces as HTTP 500.
        paymentService.processPayment(order.getTotal());
        return orderRepository.save(order);
    }

    public Order findById(Long id) {
        return orderRepository.findById(id).orElseThrow();
    }
}
